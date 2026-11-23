"""Test NARX"""

import warnings

import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose, assert_almost_equal, assert_array_equal
from sklearn.metrics import r2_score
from sklearn.utils._array_api import move_to
from sklearn.utils.estimator_checks import check_estimator, config_context

from fastcan.narx import (
    NARX,
    fd2tp,
    make_narx,
    make_poly_features,
    make_poly_ids,
    make_time_shift_features,
    make_time_shift_ids,
    print_narx,
    tp2fd,
)
from fastcan.utils import mask_missing_values


def _make_data(multi_output, nan, rng, monkeypatch, array_type="numpy"):
    if multi_output:
        n_samples = 1000
        max_delay = 3
        e0 = rng.normal(0, 0.1, n_samples)
        e1 = rng.normal(0, 0.02, n_samples)
        u0 = rng.uniform(0, 1, n_samples + max_delay)
        u1 = rng.normal(0, 0.1, n_samples + max_delay)
        y0 = np.zeros(n_samples + max_delay)
        y1 = np.zeros(n_samples + max_delay)
        for i in range(max_delay, n_samples + max_delay):
            y0[i] = (
                0.5 * y0[i - 1]
                + 0.8 * y1[i - 1]
                + 0.3 * u0[i] ** 2
                + 2 * u0[i - 1] * u0[i - 3]
                + 1.5 * u0[i - 2] * u1[i - 3]
                + 1
            )
            y1[i] = (
                0.6 * y1[i - 1]
                - 0.2 * y0[i - 1] * y1[i - 2]
                + 0.3 * u1[i] ** 2
                + 1.5 * u1[i - 2] * u0[i - 3]
                + 0.5
            )
        y = np.c_[y0[max_delay:] + e0, y1[max_delay:] + e1]
        X = np.c_[u0[max_delay:], u1[max_delay:]]
        n_outputs = 2
    else:
        rng = np.random.default_rng(12345)
        n_samples = 1000
        max_delay = 3
        e = rng.normal(0, 0.1, n_samples)
        u0 = rng.uniform(0, 1, n_samples + max_delay)
        u1 = rng.normal(0, 0.1, n_samples)
        y = np.zeros(n_samples + max_delay)
        for i in range(max_delay, n_samples + max_delay):
            y[i] = (
                0.5 * y[i - 1]
                + 0.3 * u0[i] ** 2
                + 2 * u0[i - 1] * u0[i - 3]
                + 1.5 * u0[i - 2] * u1[i - max_delay]
                + 1
            )
        y = y[max_delay:] + e
        X = np.c_[u0[max_delay:], u1]
        n_outputs = 1

    if nan:
        X_nan_ids = rng.choice(n_samples, 20, replace=False)
        y_nan_ids = rng.choice(n_samples, 10, replace=False)
        X[X_nan_ids] = np.nan
        y[y_nan_ids] = np.nan

    X = np.asfortranarray(X)
    y = np.asfortranarray(y)
    if array_type == "pytorch":
        monkeypatch.setenv("SCIPY_ARRAY_API", "1")
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.mps.is_available()
            else "cpu"
        )
        torch.set_default_device(device)
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
    return X, y, n_outputs


def test_narx_is_sklearn_estimator():
    # Skip 0 feature check for NARX, as AR models have no features
    expected_failures = {
        "check_estimators_empty_data_messages": ("NARX can handle 0 feature."),
        "check_requires_y_none": "validate_params error message mismatch",
    }
    with pytest.warns(UserWarning, match="output_ids got"):
        check_estimator(NARX(), expected_failed_checks=expected_failures)


def test_poly_ids(monkeypatch):
    with pytest.raises(ValueError, match=r"The current configuration would .*"):
        make_poly_ids(10, 1000)

    # Mock combinations_with_replacement to avoid heavy computation
    monkeypatch.setattr(
        "fastcan.narx._feature.combinations_with_replacement",
        lambda *args, **kwargs: iter([[0, 0]]),
    )
    with pytest.warns(UserWarning, match=r"Total number of polynomial features .*"):
        make_poly_ids(18, 10)


def test_time_ids():
    with pytest.raises(ValueError, match=r"The length of `include_zero_delay`.*"):
        make_time_shift_ids(3, 2, [False, True, False, True])


@pytest.mark.parametrize("array_type", ["numpy", "pytorch"])
@pytest.mark.parametrize("multi_output", [False, True])
@pytest.mark.parametrize("nan", [False, True])
def test_narx(nan, multi_output, monkeypatch, array_type):
    """Test NARX"""

    rng = np.random.default_rng(12345)
    X, y, n_outputs = _make_data(multi_output, nan, rng, monkeypatch, array_type)

    X_np = move_to(X, xp=np, device="cpu")
    y_np = move_to(y, xp=np, device="cpu")

    with config_context(array_api_dispatch=(array_type == "pytorch")):
        if multi_output:
            narx_score = make_narx(
                X,
                y,
                n_terms_to_select=[5, 4],
                max_delay=3,
                poly_degree=2,
                verbose=0,
            )
        else:
            narx_score = make_narx(
                X,
                y,
                n_terms_to_select=4,
                max_delay=3,
                poly_degree=2,
                verbose=0,
            )

    narx_score.fit(X_np, y_np)

    assert (
        r2_score(*mask_missing_values(y_np, narx_score.predict(X_np, y_init=y_np)))
        > 0.5
    )

    params: dict = {
        "n_terms_to_select": rng.integers(low=2, high=4),
        "max_delay": rng.integers(low=0, high=10),
        "poly_degree": rng.integers(low=2, high=5),
    }

    with config_context(array_api_dispatch=(array_type == "pytorch")):
        narx_default = make_narx(X=X, y=y, **params)

    if multi_output:
        assert narx_default.feat_ids.shape[0] == params["n_terms_to_select"] * 2
    else:
        assert narx_default.feat_ids.shape[0] == params["n_terms_to_select"]

    params["include_zero_delay"] = [False, True]
    with config_context(array_api_dispatch=(array_type == "pytorch")):
        narx_0_delay = make_narx(X=X, y=y, **params)
    time_shift_ids, _ = fd2tp(narx_0_delay.feat_ids, narx_0_delay.delay_ids)
    time_ids_u0 = time_shift_ids[time_shift_ids[:, 0] == 0]
    time_ids_u1 = time_shift_ids[time_shift_ids[:, 0] == 1]
    time_ids_y = time_shift_ids[time_shift_ids[:, 0] == 2]
    assert ~np.isin(0, time_ids_u0[:, 1]) or (time_ids_u0.size == 0)
    assert np.isin(0, time_ids_u1[:, 1]) or (time_ids_u1.size == 0)
    assert ~np.isin(0, time_ids_y[:, 1]) or (time_ids_y.size == 0)

    params["static_indices"] = [1]
    with config_context(array_api_dispatch=(array_type == "pytorch")):
        narx_static = make_narx(X=X, y=y, **params)
    time_shift_ids, _ = fd2tp(narx_static.feat_ids, narx_static.delay_ids)
    time_ids_u1 = time_shift_ids[time_shift_ids[:, 0] == 1]
    if time_ids_u1.size != 0:
        assert time_ids_u1[0, 1] == 0

    params["refine_drop"] = 1
    params["refine_max_iter"] = 10
    with config_context(array_api_dispatch=(array_type == "pytorch")):
        narx_drop = make_narx(X=X, y=y, **params)
    narx_drop_coef = narx_drop.fit(X_np, y_np).coef_

    time_shift_ids = make_time_shift_ids(
        X_np.shape[1] + n_outputs, 5, include_zero_delay=False
    )
    poly_ids = make_poly_ids(time_shift_ids.shape[0], 2)
    if multi_output:
        n_terms = poly_ids.shape[0]
        output_ids = [0] * n_terms
        output_ids[-1] = 1
    else:
        output_ids = None
    feat_ids, delay_ids = tp2fd(time_shift_ids, poly_ids)
    narx_osa = NARX(feat_ids=feat_ids, delay_ids=delay_ids, output_ids=output_ids).fit(
        X_np, y_np
    )
    assert narx_osa.coef_.size == poly_ids.shape[0]
    narx_osa_msa = narx_drop.fit(X_np, y_np, coef_init="one_step_ahead")
    narx_osa_msa_coef = narx_osa_msa.coef_
    narx_array_init_msa = narx_osa_msa.fit(
        X_np, y_np, coef_init=np.zeros(narx_osa_msa_coef.size + n_outputs)
    )
    assert np.any(narx_array_init_msa.coef_ != narx_drop_coef)
    assert np.any(narx_osa_msa_coef != narx_array_init_msa.coef_)

    if multi_output:
        y_init = np.ones((narx_array_init_msa.max_delay_, n_outputs), order="F")
    else:
        y_init = [1] * narx_array_init_msa.max_delay_
    y_hat = narx_array_init_msa.predict(X_np, y_init=y_init)
    assert_array_equal(y_hat[: narx_array_init_msa.max_delay_], y_init)

    with pytest.raises(ValueError, match=r"`coef_init` should have the shape of .*"):
        narx_array_init_msa.fit(X_np, y_np, coef_init=np.zeros(narx_osa_msa_coef.size))

    time_shift_ids = make_time_shift_ids(
        X_np.shape[1] + n_outputs + 1, 3, include_zero_delay=False
    )
    poly_ids = make_poly_ids(time_shift_ids.shape[0], 2)
    feat_ids, delay_ids = tp2fd(time_shift_ids, poly_ids)
    if multi_output:
        n_terms = poly_ids.shape[0]
        output_ids = [0] * n_terms
        output_ids[-1] = 1
    else:
        output_ids = None
    with pytest.raises(ValueError, match=r"The element x of feat_ids should satisfy.*"):
        narx_osa = NARX(
            feat_ids=feat_ids, delay_ids=delay_ids, output_ids=output_ids
        ).fit(X_np, y_np)

    time_shift_ids = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 1],
            [1, 2],
        ]
    )
    poly_ids = make_poly_ids(time_shift_ids.shape[0], 2)
    feat_ids, delay_ids = tp2fd(time_shift_ids, poly_ids)
    delay_ids[0, 0] = -2
    n_terms = poly_ids.shape[0]
    output_ids = [0] * n_terms
    output_ids[-1] = 1
    with pytest.raises(ValueError, match=r"The element x of delay_ids should be -1.*"):
        narx_osa = NARX(
            feat_ids=feat_ids, delay_ids=delay_ids, output_ids=output_ids
        ).fit(X_np, y_np)

    time_shift_ids = make_time_shift_ids(
        X_np.shape[1] + n_outputs, 3, include_zero_delay=False
    )
    poly_ids = make_poly_ids(time_shift_ids.shape[0], 2)
    feat_ids, delay_ids = tp2fd(time_shift_ids, poly_ids)
    delay_ids_shape_err = np.delete(delay_ids, 0, axis=0)
    n_terms = poly_ids.shape[0]
    output_ids = [0] * n_terms
    output_ids[-1] = 1
    with pytest.raises(
        ValueError, match=r"The shape of delay_ids should be equal to .*"
    ):
        narx_osa = NARX(
            feat_ids=feat_ids, delay_ids=delay_ids_shape_err, output_ids=output_ids
        ).fit(X_np, y_np)
    delay_ids_max_err = np.copy(delay_ids)
    delay_ids_max_err[0, 1] = X_np.shape[0]
    with pytest.raises(
        ValueError, match=r"The element x of delay_ids should satisfy -1.*"
    ):
        narx_osa = NARX(
            feat_ids=feat_ids, delay_ids=delay_ids_max_err, output_ids=output_ids
        ).fit(X_np, y_np)


def test_multi_output_warn():
    rng = np.random.default_rng(12345)
    X = rng.random((10, 2))
    y = rng.random((10, 2))
    for i in range(2):
        if i == 0:
            # X only, grad does not have dynamic part
            time_shift_ids = np.array([[0, 1], [1, 1]])
            poly_ids = np.array([[0, 0], [1, 1]])
        else:
            time_shift_ids = np.array([[0, 0], [1, 1], [2, 1]])
            poly_ids = np.array([[0, 0], [1, 1], [-1, 2]])
        feat_ids, delay_ids = tp2fd(time_shift_ids, poly_ids)

        narx = NARX(feat_ids=feat_ids, delay_ids=delay_ids)
        with pytest.warns(UserWarning, match="output_ids got"):
            narx.fit(X, y)
        y_pred = narx.predict(X)
        assert_almost_equal(
            np.std(y_pred[narx.max_delay_ :, 1] - np.mean(y[:, 1])), 0.0
        )

        X_nan = np.copy(X)
        y_nan = np.copy(y)
        X_nan[4, 0] = np.nan
        y_nan[4, 1] = np.nan
        for coef_init in [None, "one_step_ahead"]:
            with pytest.warns(UserWarning, match="output_ids got"):
                y_pred = narx.fit(X_nan, y_nan, coef_init=coef_init).predict(X_nan)
            y_nan_masked, y_pred_masked = mask_missing_values(y_nan, y_pred)
            y_pred_std = np.std(
                y_pred_masked[y_pred_masked[:, 0] != 0, 1] - np.mean(y_nan_masked[:, 1])
            )
            assert_almost_equal(y_pred_std, 0.0)


def test_fit_intercept():
    rng = np.random.default_rng(12345)
    X = rng.random((10, 2))
    y = rng.random((10, 1))
    time_shift_ids = np.array([[0, 1], [1, 1]])
    poly_ids = np.array([[0, 0], [1, 1]])
    feat_ids, delay_ids = tp2fd(time_shift_ids, poly_ids)

    narx = NARX(
        feat_ids=feat_ids,
        delay_ids=delay_ids,
        fit_intercept=False,
    )
    narx.fit(X, y)
    assert_almost_equal(narx.intercept_, 0.0)
    narx.fit(X, y, coef_init="one_step_ahead")
    assert_almost_equal(narx.intercept_, 0.0)

    X = rng.random((10, 2))
    y = rng.random((10, 2))
    time_shift_ids = np.array([[0, 1], [1, 1]])
    poly_ids = np.array([[0, 0], [1, 1]])
    feat_ids, delay_ids = tp2fd(time_shift_ids, poly_ids)

    narx = make_narx(X, y, 1, 2, 2, fit_intercept=False)
    narx.fit(X, y)
    assert_array_equal(narx.intercept_, [0.0, 0.0])
    narx.fit(X, y, coef_init="one_step_ahead")
    assert_array_equal(narx.intercept_, [0.0, 0.0])

    narx = NARX(
        feat_ids=feat_ids,
        delay_ids=delay_ids,
        fit_intercept=False,
    )
    with pytest.warns(UserWarning, match="output_ids got"):
        narx.fit(X, y)
    assert_array_equal(narx.intercept_, [0.0, 0.0])
    with pytest.warns(UserWarning, match="output_ids got"):
        narx.fit(X, y, coef_init=[0, 0])
    assert_array_equal(narx.intercept_, [0.0, 0.0])


def test_multi_output_error():
    rng = np.random.default_rng(12345)
    X = rng.random((10, 2))
    y = rng.random((10, 2))
    time_shift_ids = np.array([[0, 1], [1, 1]])
    poly_ids = np.array([[0, 0], [1, 1]])
    feat_ids, delay_ids = tp2fd(time_shift_ids, poly_ids)

    narx = NARX(
        feat_ids=feat_ids,
        delay_ids=delay_ids,
        output_ids=[0],
    )
    with pytest.raises(ValueError, match="The length of output_ids should"):
        narx.fit(X, y)

    narx = NARX(
        feat_ids=feat_ids,
        delay_ids=delay_ids,
        output_ids=[0, 2],
    )
    with pytest.raises(
        ValueError, match=r"The element x of output_ids should satisfy 0 <=.*"
    ):
        narx.fit(X, y)

    with pytest.raises(ValueError, match="The length of `n_terms_to_select` should"):
        make_narx(X=X, y=y, n_terms_to_select=[2], max_delay=3, poly_degree=2)

    narx = make_narx(X=X, y=y, n_terms_to_select=[2, 2], max_delay=3, poly_degree=2)
    narx.fit(X, y)
    with pytest.raises(ValueError, match="`y_init` should have "):
        narx.predict(X, y_init=[1, 1, 1])

    narx = NARX(
        feat_ids=np.array([[0, 1], [-1, -1]]),
        delay_ids=np.array([[0, 1], [-1, -1]]),
        output_ids=[0, 1],
    )
    with pytest.raises(ValueError, match=r"`feat_ids` should not contain rows that.*"):
        narx.fit(X, y)


def test_sample_weight():
    rng = np.random.default_rng(12345)
    n_samples = 100
    max_delay = 3
    e = rng.normal(0, 0.1, n_samples)
    u0 = rng.uniform(0, 1, n_samples + max_delay)
    u1 = rng.normal(0, 0.1, n_samples)
    y = np.zeros(n_samples + max_delay)
    for i in range(max_delay, n_samples + max_delay):
        y[i] = (
            0.5 * y[i - 1]
            + 0.3 * u0[i] ** 2
            + 2 * u0[i - 1] * u0[i - 3]
            + 1.5 * u0[i - 2] * u1[i - max_delay]
            + 1
        )
    y = y[max_delay:] + e
    X = np.c_[u0[max_delay:], u1]

    sample_weight = np.ones(n_samples)
    sample_weight[:10] = 0  # Set the first 10 samples to have zero weight

    narx = make_narx(X=X, y=y, n_terms_to_select=3, max_delay=3, poly_degree=2)
    narx.fit(X, y, sample_weight=sample_weight)
    coef_w = narx.coef_
    narx.fit(X, y)
    coef_ = narx.coef_

    assert np.any(coef_w != coef_)

    X = np.array(
        [
            [1, 1],
            [1, 2],
            [1, 3],
            [1, 4],
            [2, 1],
            [2, 2],
            [2, 3],
            [2, 4],
            [3, 1],
            [3, 2],
            [3, 3],
            [3, 4],
        ]
    )
    y = np.array([1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 2, 2])
    sw = rng.integers(0, 5, size=12)
    X_repeated = np.repeat(X, sw, axis=0)
    y_repeated = np.repeat(y, sw)
    narx_osa = NARX().fit(X_repeated, y_repeated)
    narx_no_sw = NARX().fit(X_repeated, y_repeated, coef_init=[0] * 3)
    assert_allclose(
        np.r_[narx_osa.coef_, narx_osa.intercept_],
        np.r_[narx_no_sw.coef_, narx_no_sw.intercept_],
    )
    narx_sw = NARX().fit(X, y, sample_weight=sw, coef_init=[0] * 3)
    assert_allclose(
        np.r_[narx_no_sw.coef_, narx_no_sw.intercept_],
        np.r_[narx_sw.coef_, narx_sw.intercept_],
    )


def test_divergence():
    # Test divergence of NARX model
    rng = np.random.default_rng(12345)
    n_samples = 100
    max_delay = 3
    e = rng.normal(0, 0.1, n_samples)
    u0 = rng.uniform(0, 1, n_samples + max_delay)
    u1 = rng.normal(0, 0.1, n_samples)
    y = np.zeros(n_samples + max_delay)
    for i in range(max_delay, n_samples + max_delay):
        y[i] = (
            0.5 * y[i - 1]
            + 0.3 * u0[i] ** 2
            + 2 * u0[i - 1] * u0[i - 3]
            + 1.5 * u0[i - 2] * u1[i - max_delay]
            + 1
        )
    y = y[max_delay:] + e
    X = np.c_[u0[max_delay:], u1]
    narx = make_narx(X, y, 3, 3, 2)
    narx.fit(X, y, coef_init=[-10, 0, 0, 0])
    y_hat = narx.predict(X, y)
    div_idx = np.where(np.abs(y_hat) > 1e20)[0][0] + 1
    assert np.all(y_hat[div_idx:] == 0)


def test_tp2fd():
    time_shift_ids = np.array(
        [
            [0, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [1, 2, 0],
        ]
    )
    poly_ids = make_poly_ids(time_shift_ids.shape[0], 2)
    with pytest.raises(ValueError, match=r"time_shift_ids should have shape.*"):
        _, _ = tp2fd(time_shift_ids, poly_ids)
    time_shift_ids = np.array(
        [
            [0, 0],
            [-1, 1],
            [1, 1],
            [1, 2],
        ]
    )
    with pytest.raises(ValueError, match=r"The element x of the first column of tim.*"):
        _, _ = tp2fd(time_shift_ids, poly_ids)
    time_shift_ids = np.array(
        [
            [0, 0],
            [0, -1],
            [1, 1],
            [1, 2],
        ]
    )
    with pytest.raises(ValueError, match=r"The element x of the second column of ti.*"):
        _, _ = tp2fd(time_shift_ids, poly_ids)
    time_shift_ids = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 1],
            [1, 2],
        ]
    )
    poly_ids[-1][-1] = 5
    with pytest.raises(ValueError, match=r"The element x of poly_ids should.*"):
        _, _ = tp2fd(time_shift_ids, poly_ids)


def test_print_narx(capsys):
    rng = np.random.default_rng(12345)
    X = rng.random((10, 2))
    y = rng.random((10, 2))
    feat_ids = np.array([[0, 1], [1, 2]])
    delay_ids = np.array([[1, 0], [2, 2]])

    narx = NARX(
        feat_ids=feat_ids,
        delay_ids=delay_ids,
        output_ids=[0, 1],
    )
    narx.fit(X, y)
    print_narx(narx)
    captured = capsys.readouterr()
    # Check if the header is present in the output
    assert "| yid |        Term        |   Coef   |" in captured.out
    # Check if the separator line is present
    assert "|-----|--------------------|----------|" in captured.out
    # Check if the intercept line for yid 0 is present
    assert "|  0  |     Intercept      |" in captured.out
    # Check if the intercept line for yid 1 is present
    assert "|  1  |     Intercept      |" in captured.out
    # Check if the term line for yid 0 is present
    assert "|  0  |  X[k-1,0]*X[k,1]   |" in captured.out
    # Check if the term line for yid 1 is present
    assert "|  1  |X[k-2,1]*y_hat[k-2,0]|" in captured.out


def test_make_narx_refine_print(capsys):
    rng = np.random.default_rng(12345)
    X = rng.random((10, 2))
    y = rng.random((10, 2))
    _ = make_narx(
        X,
        y,
        n_terms_to_select=2,
        max_delay=2,
        poly_degree=2,
        refine_drop=1,
    )
    captured = capsys.readouterr()
    assert "No. of iterations: " in captured.out


@pytest.mark.parametrize("array_type", ["numpy", "pytorch"])
def test_make_narx_lazy(array_type, monkeypatch):
    """Test lazy selection in make_narx."""
    rng = np.random.default_rng(12345)
    X, y, _ = _make_data(False, False, rng, monkeypatch, array_type)
    X_np = move_to(X, xp=np, device="cpu")
    y_np = move_to(y, xp=np, device="cpu")
    max_delay = 3
    poly_degree = 3
    session_sizes = [200, 300, 500]
    with config_context(array_api_dispatch=(array_type == "pytorch")):
        narx_normal = make_narx(
            X,
            y,
            n_terms_to_select=3,
            max_delay=max_delay,
            poly_degree=poly_degree,
            session_sizes=session_sizes,
            lazy=False,
        )
        narx_lazy = make_narx(
            X,
            y,
            n_terms_to_select=3,
            max_delay=max_delay,
            poly_degree=poly_degree,
            session_sizes=session_sizes,
            lazy=True,
            batch_size=5,
        )
    narx_normal.fit(X_np, y_np)
    narx_lazy.fit(X_np, y_np)
    assert_array_equal(narx_normal.feat_ids, narx_lazy.feat_ids)
    assert_array_equal(narx_normal.delay_ids, narx_lazy.delay_ids)


def test_make_narx_max_candidates():
    """Test max_candidates and random_state in make_narx."""
    rng = np.random.default_rng(12345)
    X = rng.random((100, 2))
    y = rng.random((100, 1))
    max_delay = 3
    poly_degree = 10
    n_terms_to_select = 5
    max_candidates = 20

    # With the same random_state, the results should be identical
    narx1 = make_narx(
        X,
        y,
        n_terms_to_select=n_terms_to_select,
        max_delay=max_delay,
        poly_degree=poly_degree,
        max_candidates=max_candidates,
        random_state=123,
        verbose=0,
    )
    narx2 = make_narx(
        X,
        y,
        n_terms_to_select=n_terms_to_select,
        max_delay=max_delay,
        poly_degree=poly_degree,
        max_candidates=max_candidates,
        random_state=123,
        verbose=0,
    )
    assert_array_equal(narx1.feat_ids, narx2.feat_ids)
    assert_array_equal(narx1.delay_ids, narx2.delay_ids)

    # With different random_state, the results should be different
    narx3 = make_narx(
        X,
        y,
        n_terms_to_select=n_terms_to_select,
        max_delay=max_delay,
        poly_degree=poly_degree,
        max_candidates=max_candidates,
        random_state=456,
        verbose=0,
    )
    assert not np.array_equal(narx1.feat_ids, narx3.feat_ids)

    # Check if number of selected terms is correct
    assert narx1.feat_ids.shape[0] == n_terms_to_select


@pytest.mark.parametrize("max_delay", [1, 3, 7, 10])
def test_nan_split(max_delay):
    rng = np.random.default_rng(12345)
    n_sessions = 10
    n_samples_per_session = 100
    X = rng.random((n_samples_per_session, 2))
    y = rng.random((n_samples_per_session, 2))
    for _ in range(n_sessions - 1):
        X = np.r_[
            X,
            [[np.nan, np.nan]] * max_delay,
            rng.random((n_samples_per_session, 2)),
        ]
        y = np.r_[
            y,
            [[np.nan, np.nan]] * max_delay,
            rng.random((n_samples_per_session, 2)),
        ]
    narx = make_narx(
        X,
        y,
        n_terms_to_select=10,
        max_delay=max_delay,
        poly_degree=3,
        verbose=0,
    ).fit(
        X,
        y,
    )

    xy_hstack = np.c_[X, y]
    time_shift_ids, poly_ids = fd2tp(narx.feat_ids_, narx.delay_ids_)
    time_shift_vars = make_time_shift_features(xy_hstack, time_shift_ids)
    poly_terms = make_poly_features(time_shift_vars, poly_ids)
    poly_terms_masked, y_masked = mask_missing_values(poly_terms, y)
    assert poly_terms_masked.shape[0] == y_masked.shape[0]
    assert poly_terms_masked.shape[0] == n_sessions * (
        n_samples_per_session - narx.max_delay_
    )


def test_default_narx_handles_zero_features():
    """Check that default NARX handles X with 0 features without error."""
    rng = np.random.default_rng(12345)
    X = np.empty((10, 0))
    y = rng.random((10, 1))
    NARX().fit(X, y)


def test_auto_reg():
    """Test auto-regression with NARX"""
    rng = np.random.default_rng(12345)
    n_samples = 100
    max_delay = 2
    e0 = rng.normal(0, 0.01, n_samples)
    e1 = rng.normal(0, 0.01, n_samples)
    y0 = np.ones(n_samples + max_delay)
    y1 = np.ones(n_samples + max_delay)
    for i in range(max_delay, n_samples + max_delay):
        y0[i] = 0.5 * y0[i - 1] + 0.8 * y1[i - 1] + 1
        y1[i] = 0.6 * y1[i - 1] - 0.2 * y0[i - 1] * y1[i - 2] + 0.5
    y = np.c_[y0[max_delay:] + e0, y1[max_delay:] + e1]
    X = np.empty((n_samples, 0))  # No features, only auto-regression

    model = make_narx(
        X,
        y,
        n_terms_to_select=2,
        max_delay=max_delay,
        poly_degree=2,
        verbose=0,
    )
    model.fit(X, y)
    y_pred = model.predict(X, y_init=y[: model.max_delay_])
    assert r2_score(y, y_pred) > 0.5

    model = make_narx(
        None,
        y,
        n_terms_to_select=2,
        max_delay=max_delay,
        poly_degree=2,
        verbose=0,
    )
    model.fit(None, y)
    y_pred = model.predict(len(y), y_init=y[: model.max_delay_])
    assert r2_score(y, y_pred) > 0.5


def test_auto_reg_error():
    rng = np.random.default_rng(12345)
    X = np.empty((10, 1))
    y = rng.random((10, 1))
    model = NARX().fit(X, y)
    with pytest.raises(ValueError, match=r"X should be an array-like of shape.*"):
        model.predict(len(y), y_init=y[: model.max_delay_])


def test_predict_ndim():
    """Test the ndim of predict output"""
    rng = np.random.default_rng(12345)
    X = rng.random((10, 2))
    y = rng.random(10)
    time_shift_ids = np.array([[0, 1], [1, 1]])
    poly_ids = np.array([[0, 0], [1, 1]])
    feat_ids, delay_ids = tp2fd(time_shift_ids, poly_ids)

    model = NARX(
        feat_ids=feat_ids,
        delay_ids=delay_ids,
        output_ids=None,
    )
    model.fit(X, y)
    y_hat = model.predict(X, y_init=y[: model.max_delay_])
    assert y_hat.ndim == 1

    model.fit(X, y.reshape(-1, 1))
    y_hat = model.predict(X, y_init=y[: model.max_delay_])
    assert y_hat.ndim == 2


def test_session_sizes():
    """Test that session sizes in make_narx and NARX.fit and error messages."""
    rng = np.random.default_rng(12345)
    X = rng.random((100, 2))
    y = rng.random((100, 1))
    with pytest.raises(ValueError, match=r"The sum of session_sizes should be.*"):
        model = make_narx(
            X,
            y,
            n_terms_to_select=5,
            max_delay=3,
            poly_degree=2,
            verbose=0,
            session_sizes=[10, 20],
        )

    model = make_narx(
        X,
        y,
        n_terms_to_select=5,
        max_delay=3,
        poly_degree=2,
        verbose=0,
    )
    with pytest.raises(
        ValueError, match=r"All elements of session_sizes should be positive.*"
    ):
        model.fit(X, y, session_sizes=[-10, 110])

    model = make_narx(
        X,
        y,
        n_terms_to_select=5,
        max_delay=3,
        poly_degree=2,
        verbose=0,
    )
    session_sizes = [20, 30, 50]
    session_sizes_cumsum = np.cumsum(session_sizes)
    model.fit(X, y, session_sizes=session_sizes)
    coef_session = model.coef_
    X_missing = X[: session_sizes_cumsum[0]]
    y_missing = y[: session_sizes_cumsum[0]]
    for i in range(len(session_sizes) - 1):
        X_missing = np.r_[
            X_missing,
            [[np.nan, np.nan]] * model.max_delay_,
            X[session_sizes_cumsum[i] : session_sizes_cumsum[i + 1]],
        ]
        y_missing = np.r_[
            y_missing,
            [[np.nan]] * model.max_delay_,
            y[session_sizes_cumsum[i] : session_sizes_cumsum[i + 1]],
        ]
    coef_missing = model.fit(X_missing, y_missing).coef_
    # Exactly same for one-step-ahead fitting
    assert np.sum(np.abs(coef_session - coef_missing)) == 0
    coef_session_msa = model.fit(
        X, y, session_sizes=session_sizes, coef_init="one_step_ahead"
    ).coef_
    coef_missing_msa = model.fit(X_missing, y_missing, coef_init="one_step_ahead").coef_
    # Exactly same for multi-step-ahead fitting
    assert np.sum(np.abs(coef_session_msa - coef_missing_msa)) == 0

    # Auto-regression with session sizes
    arm = make_narx(
        X=None,
        y=y,
        n_terms_to_select=5,
        max_delay=3,
        poly_degree=2,
        verbose=0,
    )
    arm.fit(X=None, y=y, session_sizes=session_sizes)
    ar_coef_session = arm.coef_
    arm.fit(X=None, y=y_missing)
    ar_coef_missing = arm.coef_
    # Exactly same for one-step-ahead fitting
    assert np.sum(np.abs(ar_coef_session - ar_coef_missing)) == 0

    arm.fit(X=None, y=y, session_sizes=session_sizes, coef_init="one_step_ahead")
    ar_coef_session_msa = arm.coef_
    arm.fit(X=None, y=y_missing, coef_init="one_step_ahead")
    ar_coef_missing_msa = arm.coef_
    # Not same for multi-step-ahead fitting
    assert np.sum(np.abs(ar_coef_session_msa - ar_coef_missing_msa)) != 0


def _short_session_data():
    """Sessions [10, 2 (shorter than max_delay=3), 8]."""
    rng = np.random.default_rng(7)
    session_sizes = [10, 2, 8]
    n_samples = sum(session_sizes)
    X = rng.random((n_samples, 2))
    y = rng.random(n_samples)
    return rng, X, y, session_sizes


def test_fit_short_session_bookkeeping():
    """fit leaves per-session valid counts and exclusion reasons."""
    _, X, y, session_sizes = _short_session_data()
    feat_ids = np.array([[1], [1], [0]], dtype=np.int32)
    delay_ids = np.array([[1], [3], [0]], dtype=np.int32)
    narx = NARX(feat_ids=feat_ids, delay_ids=delay_ids)
    with pytest.warns(UserWarning, match="no valid sample"):
        narx.fit(X, y, session_sizes=session_sizes)

    assert_array_equal(narx.session_sizes_, session_sizes)
    assert_array_equal(narx.n_valid_samples_, [7, 0, 5])
    assert_array_equal(narx.n_excluded_samples_, [3, 2, 3])
    assert narx.exclusion_reasons_[1] == {"warmup": 2, "non_finite": 0}
    # the short session contributes no row to the fit
    assert narx.n_valid_samples_.sum() == 12


def test_fit_short_session_non_finite_reason():
    """A non-finite y row outside the warmup region is reported separately."""
    _, X, y, session_sizes = _short_session_data()
    y = y.copy()
    y[6] = np.nan
    feat_ids = np.array([[1], [1], [0]], dtype=np.int32)
    delay_ids = np.array([[1], [3], [0]], dtype=np.int32)
    with pytest.warns(UserWarning):
        narx = NARX(feat_ids=feat_ids, delay_ids=delay_ids).fit(
            X, y, session_sizes=session_sizes
        )
    assert narx.exclusion_reasons_[0] == {"warmup": 3, "non_finite": 1}
    assert narx.n_valid_samples_[0] == 6


@pytest.mark.parametrize("lazy", [False, True])
def test_make_narx_short_session_report(lazy):
    """Candidate selection keeps the same per-session bookkeeping on both paths."""
    _, X, y, session_sizes = _short_session_data()
    kwargs = {
        "n_terms_to_select": 3,
        "max_delay": 3,
        "poly_degree": 2,
        "session_sizes": session_sizes,
    }

    def build(**extra):
        if lazy:
            return make_narx(X, y, lazy=True, batch_size=5, **kwargs, **extra)
        return make_narx(X, y, verbose=0, **kwargs, **extra)

    with pytest.warns(UserWarning, match="no valid sample"):
        narx, report = build(return_report=True)
    assert_array_equal(report["session_sizes"], session_sizes)
    assert_array_equal(report["n_valid_samples"], [7, 0, 5])
    assert_array_equal(report["n_excluded_samples"], [3, 2, 3])
    assert report["exclusion_reasons"][1] == {"warmup": 2, "non_finite": 0}
    # the report is also attached to the unfitted NARX
    assert narx.session_report_ == report
    # default return value stays a bare NARX
    with pytest.warns(UserWarning):
        bare = build()
    assert isinstance(bare, NARX)


def test_predict_short_session_nan_slots():
    """predict returns NaN slots aligned with the request for short sessions."""
    rng, X, y, session_sizes = _short_session_data()
    feat_ids = np.array([[1], [1], [0]], dtype=np.int32)
    delay_ids = np.array([[1], [3], [0]], dtype=np.int32)
    narx = NARX(feat_ids=feat_ids, delay_ids=delay_ids)
    with pytest.warns(UserWarning):
        narx.fit(X, y, session_sizes=session_sizes)

    Xp = rng.random((X.shape[0], 2))
    y_init = rng.random((narx.max_delay_, 1))
    y_hat = narx.predict(Xp, y_init=y_init, session_sizes=session_sizes)
    assert y_hat.shape == (X.shape[0],)
    # the whole short session is NaN
    assert np.all(np.isnan(y_hat[10:12]))
    # warmup rows of the other sessions are NaN as well
    assert np.all(np.isnan(y_hat[0:3]))
    assert np.all(np.isnan(y_hat[12:15]))
    # every other row is a finite prediction
    assert np.all(np.isfinite(np.delete(y_hat, [0, 1, 2, 10, 11, 12, 13, 14])))


def test_predict_short_session_does_not_shift_next_session():
    """A short session must not mismatch the start of the next session."""
    rng, X, y, session_sizes = _short_session_data()
    feat_ids = np.array([[1], [1], [0]], dtype=np.int32)
    delay_ids = np.array([[1], [3], [0]], dtype=np.int32)
    narx = NARX(feat_ids=feat_ids, delay_ids=delay_ids)
    with pytest.warns(UserWarning):
        narx.fit(X, y, session_sizes=session_sizes)

    Xp = rng.random((X.shape[0], 2))
    y_init = rng.random((narx.max_delay_, 1))
    y_hat = narx.predict(Xp, y_init=y_init, session_sizes=session_sizes)
    # The last session predicted standalone must restart from y_init[0] and
    # therefore match the same rows of the multi-session prediction.
    y_hat_standalone = narx.predict(Xp[12:20], y_init=y_init)
    assert_allclose(y_hat[15:20], y_hat_standalone[3:], rtol=1e-12, atol=1e-12)


def test_predict_without_session_sizes_keeps_warmup_values():
    """Backward compatibility: single-session predict keeps y_init warmup rows."""
    rng = np.random.default_rng(11)
    X = rng.random((20, 1))
    y = rng.random(20)
    feat_ids = np.array([[1], [0]], dtype=np.int32)
    delay_ids = np.array([[1], [0]], dtype=np.int32)
    narx = NARX(feat_ids=feat_ids, delay_ids=delay_ids).fit(X, y)
    y_init = rng.random((narx.max_delay_, 1))
    y_hat = narx.predict(X, y_init=y_init)
    assert_allclose(y_hat[: narx.max_delay_], y_init.ravel())
    assert np.all(np.isfinite(y_hat))


def test_fit_session_zero_weight_reason():
    """Zero-weight rows outside warmup are reported with their own reason."""
    _, X, y, session_sizes = _short_session_data()
    feat_ids = np.array([[1], [0]], dtype=np.int32)
    delay_ids = np.array([[1], [0]], dtype=np.int32)
    sample_weight = np.ones(X.shape[0])
    # row 5 is a valid (non-warmup) row of the first session
    sample_weight[5] = 0.0
    narx = NARX(feat_ids=feat_ids, delay_ids=delay_ids).fit(
        X,
        y,
        sample_weight=sample_weight,
        session_sizes=session_sizes,
    )
    reason0 = narx.exclusion_reasons_[0]
    assert reason0["zero_weight"] == 1
    assert narx.exclusion_reasons_[2]["zero_weight"] == 0
    # session 0 has 9 valid rows with max_delay=1, one is zero-weighted
    assert narx.n_valid_samples_[0] == 8


def test_predict_auto_regressive_short_session():
    """Integer-step AR predict handles short sessions with NaN slots."""
    rng = np.random.default_rng(5)
    n_samples = 20
    max_delay = 2
    y = np.zeros(n_samples)
    for i in range(1, n_samples):
        y[i] = 0.6 * y[i - 1] + rng.normal(0, 0.05)
    session_sizes = [10, 2, 8]

    arm = make_narx(
        None,
        y,
        n_terms_to_select=2,
        max_delay=max_delay,
        poly_degree=1,
        verbose=0,
    )
    with pytest.warns(UserWarning, match="no valid sample"):
        arm.fit(None, y, session_sizes=session_sizes)
    assert_array_equal(arm.n_valid_samples_, [8, 0, 6])

    y_init = y[:max_delay].reshape(-1, 1)
    y_hat = arm.predict(n_samples, y_init=y_init, session_sizes=session_sizes)
    assert y_hat.shape == (n_samples,)
    assert np.all(np.isnan(y_hat[10:12]))
    assert np.all(np.isnan(y_hat[0:2]))
    assert np.all(np.isnan(y_hat[12:14]))
    assert np.all(np.isfinite(np.delete(y_hat, [0, 1, 10, 11, 12, 13])))
    # the last session is not shifted by the short session before it
    standalone = arm.predict(8, y_init=y_init)
    assert_allclose(y_hat[14:20], standalone[2:], rtol=1e-12, atol=1e-12)


def test_make_narx_missing_propagated_reason():
    """A y NaN excludes its row plus rows whose delayed terms inherit it."""
    rng = np.random.default_rng(9)
    X = rng.random((30, 2))
    y = rng.random(30)
    y[10] = np.nan
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        _, report = make_narx(
            X,
            y,
            n_terms_to_select=3,
            max_delay=3,
            poly_degree=2,
            return_report=True,
            verbose=0,
        )
    assert not any("no valid sample" in str(w.message) for w in record)
    reason = report["exclusion_reasons"][0]
    assert reason["non_finite"] == 1
    assert reason["missing_propagated"] == 3
    assert reason["warmup"] == 3
    assert report["n_valid_samples"][0] == 23
