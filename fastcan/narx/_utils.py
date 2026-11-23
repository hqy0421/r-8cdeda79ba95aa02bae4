"""NARX-related utilities."""

# Authors: The fastcan developers
# SPDX-License-Identifier: MIT

import warnings
from functools import partial
from numbers import Integral

import numpy as np
from scipy.stats import rankdata
from sklearn.utils import (
    check_array,
    check_consistent_length,
    column_or_1d,
)
from sklearn.utils._array_api import (
    get_namespace_and_device,
    move_to,
    supported_float_dtypes,
)
from sklearn.utils._param_validation import Interval, StrOptions, validate_params
from sklearn.utils.validation import check_is_fitted

from .._fastcan import FastCan
from .._lazyfastcan import LazyFastCan, _skip2valid
from .._refine import refine
from ..utils import mask_missing_values
from ._base import (
    NARX,
    _prepare_poly_terms,
    _session_warmup_summary,
    _summarize_session_masks,
    _validate_session_sizes,
)
from ._feature import (
    _make_a_time_shift_feature,
    make_poly_ids,
    make_time_shift_ids,
    tp2fd,
)


@validate_params(
    {
        "narx": [NARX],
        "term_space": [Interval(Integral, 1, None, closed="left")],
        "coef_space": [Interval(Integral, 1, None, closed="left")],
        "float_precision": [Interval(Integral, 0, None, closed="left")],
    },
    prefer_skip_nested_validation=True,
)
def print_narx(
    narx,
    term_space=20,
    coef_space=10,
    float_precision=3,
):
    """Print a NARX model as a Table which contains y idx, Term, and Coef.
    y idx is used to indicate which output (assuming it is a multi-output case)
    the term belongs to.

    Parameters
    ----------
    narx : NARX model
        The NARX model to be printed.

    term_space: int, default=20
        The space for the column of Term.

    coef_space: int, default=10
        The space for the column of Coef.

    float_precision: int, default=3
        The number of places after the decimal for Coef.

    Returns
    -------
    table : str
        The table of output index, terms, and coefficients of the NARX model.

    Examples
    --------
    >>> from sklearn.datasets import load_diabetes
    >>> from fastcan.narx import print_narx, NARX
    >>> X, y = load_diabetes(return_X_y=True)
    >>> print_narx(NARX().fit(X, y), term_space=10, coef_space=5, float_precision=0)
    | yid |   Term   |Coef |
    |-----|----------|-----|
    |  0  |Intercept | 152 |
    |  0  |  X[k,0]  | -10 |
    |  0  |  X[k,1]  |-240 |
    |  0  |  X[k,2]  | 520 |
    |  0  |  X[k,3]  | 324 |
    |  0  |  X[k,4]  |-792 |
    |  0  |  X[k,5]  | 477 |
    |  0  |  X[k,6]  | 101 |
    |  0  |  X[k,7]  | 177 |
    |  0  |  X[k,8]  | 751 |
    |  0  |  X[k,9]  | 68  |
    """
    check_is_fitted(narx)

    def _get_term_str(term_feat_ids, term_delay_ids):
        term_str = ""
        for _, (feat_id, delay_id) in enumerate(zip(term_feat_ids, term_delay_ids)):
            if -1 < feat_id < narx.n_features_in_:
                if delay_id == 0:
                    term_str += f"*X[k,{feat_id}]"
                else:
                    term_str += f"*X[k-{delay_id},{feat_id}]"
            elif feat_id >= narx.n_features_in_:
                term_str += f"*y_hat[k-{delay_id},{feat_id - narx.n_features_in_}]"
        return term_str[1:]

    yid_space = 5
    print(
        f"|{'yid':^{yid_space}}"
        + f"|{'Term':^{term_space}}"
        + f"|{'Coef':^{coef_space}}|"
    )
    print(f"|{'-' * yid_space}|{'-' * term_space}|{'-' * coef_space}|")
    for i in range(narx.n_outputs_):
        print(
            f"|{i:^{yid_space}}|"
            + f"{'Intercept':^{term_space}}|"
            + f"{narx.intercept_[i]:^{coef_space}.{float_precision}f}|"
        )
    for i, term_id in enumerate(zip(narx.feat_ids_, narx.delay_ids_)):
        print(
            f"|{narx.output_ids_[i]:^{yid_space}}|"
            + f"{_get_term_str(*term_id):^{term_space}}|"
            + f"{narx.coef_[i]:^{coef_space}.{float_precision}f}|"
        )


@validate_params(
    {
        "X": ["array-like", None],
        "y": ["array-like"],
        "n_terms_to_select": [
            Interval(Integral, 1, None, closed="left"),
            "array-like",
        ],
        "max_delay": [
            Interval(Integral, 0, None, closed="left"),
        ],
        "poly_degree": [
            Interval(Integral, 1, None, closed="left"),
        ],
        "fit_intercept": ["boolean"],
        "session_sizes": [None, "array-like"],
        "max_candidates": [None, Interval(Integral, 1, None, closed="left")],
        "random_state": ["random_state"],
        "include_zero_delay": [None, "array-like"],
        "static_indices": [None, "array-like"],
        "lazy": ["boolean"],
        "batch_size": [
            Interval(Integral, 1, None, closed="left"),
        ],
        "refine_verbose": ["verbose"],
        "refine_drop": [
            None,
            Interval(Integral, 1, None, closed="left"),
            StrOptions({"all"}),
        ],
        "refine_max_iter": [
            None,
            Interval(Integral, 1, None, closed="left"),
        ],
        "return_report": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
def make_narx(
    X,
    y,
    n_terms_to_select,
    max_delay=1,
    poly_degree=1,
    *,
    fit_intercept=True,
    session_sizes=None,
    max_candidates=None,
    random_state=None,
    include_zero_delay=None,
    static_indices=None,
    lazy=False,
    batch_size=16,
    refine_verbose=1,
    refine_drop=None,
    refine_max_iter=None,
    return_report=False,
    **params,
):
    """Find `time_shift_ids`, `poly_ids`, `output_ids` for a NARX model.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.

    y : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Target vector or matrix.

    n_terms_to_select : int or array-like of shape (n_outputs,)
        The parameter is the absolute number of polynomial terms to select for
        each output. If `n_terms_to_select` is an integer, it is the
        same for all outputs.

    max_delay : int, default=1
        The maximum delay of time shift features.

    poly_degree : int, default=1
        The maximum degree of polynomial features.

    fit_intercept : bool, default=True
        Whether to fit the intercept. If set to False, intercept will be zeros.

    session_sizes : array-like of shape (n_sessions,), default=None
        The sizes of measurement sessions for time-series.
        The sum of session_sizes should be equal to n_samples.
        If None, the whole data is treated as one session.

        .. versionadded:: 0.5.0

    max_candidates : int, default=None
        Maximum number of candidate polynomial terms retained before selection.
        Randomly selected by reservoir sampling.
        If None, all candidates are considered.

    random_state : int or RandomState instance, default=None
        Used when `max_candidates` is not None to subsample candidate terms.
        See :term:`Glossary <random_state>` for details.

    include_zero_delay : {None, array-like} of shape (n_features,) default=None
        Whether to include the original (zero-delay) features.

    static_indices : {None, array-like} of shape (n_static_features,) default=None
        The indices of static features without time delay.

        .. note::
            If the corresponding include_zero_delay of the static features is False, the
            static feature will be excluded from candidate features.

    lazy : bool, default=False
        Whether to use LazyFastCan for selection. If False, FastCan will be used.

        .. versionadded:: 0.6.0

    batch_size : int, default=16
        The batch size of polynomial time shift terms generated at a time.
        Only available when `lazy` is True.

        .. versionadded:: 0.6.1

    refine_verbose : int, default=1
        The verbosity level of refine. Only available when `lazy` is False.

    refine_drop : int or "all", default=None
            The number of the selected features dropped for the consequent
            reselection. If `drop` is None, no refining will be performed.
            Only available when `lazy` is False.

    refine_max_iter : int, default=None
        The maximum number of valid iterations in the refining process.
        Only available when `lazy` is False.

    return_report : bool, default=False
        If True, additionally return a report recording, for each measurement
        session, the number of samples retained for candidate selection and
        the number of samples excluded together with the reason (``"warmup"``
        or ``"non_finite"``). Sessions shorter than `max_delay` are reported
        with zero retained sample instead of being silently discarded.

        .. versionadded:: 0.7.0

    **params : dict
            Keyword arguments passed to
            `fastcan.FastCan`.
            Only available when `lazy` is False.

    Returns
    -------
    narx : NARX
        NARX instance. When `return_report` is True, a tuple
        ``(narx, report)`` is returned, where ``report`` is a dict with keys
        ``session_sizes``, ``n_valid_samples``, ``n_excluded_samples`` and
        ``exclusion_reasons``. The same bookkeeping is also stored on the
        returned NARX as the ``session_report_`` attribute.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.metrics import mean_squared_error
    >>> from fastcan.narx import make_narx, print_narx
    >>> rng = np.random.default_rng(12345)
    >>> n_samples = 1000
    >>> max_delay = 3
    >>> e = rng.normal(0, 0.1, n_samples)
    >>> u0 = rng.uniform(0, 1, n_samples+max_delay) # input
    >>> u1 = rng.normal(0, 0.1, n_samples) # static input (i.e. no time shift)
    >>> y = np.zeros(n_samples+max_delay) # output
    >>> for i in range(max_delay, n_samples+max_delay):
    ...     y[i] = (0.5*y[i-1] + 0.3*u0[i]**2 + 2*u0[i-1]*u0[i-3] +
    ...             1.5*u0[i-2]*u1[i-max_delay] + 1)
    >>> y = y[max_delay:]+e
    >>> X = np.c_[u0[max_delay:], u1]
    >>> narx = make_narx(X=X,
    ...     y=y,
    ...     n_terms_to_select=4,
    ...     max_delay=3,
    ...     poly_degree=2,
    ...     static_indices=[1],
    ...     eta=True,
    ...     verbose=0,
    ...     refine_verbose=0,
    ...     refine_drop=1)
    >>> print(f"{mean_squared_error(y, narx.fit(X, y).predict(X)):.4f}")
    0.0289
    >>> print_narx(narx)
    | yid |        Term        |   Coef   |
    |-----|--------------------|----------|
    |  0  |     Intercept      |  1.050   |
    |  0  |    y_hat[k-1,0]    |  0.484   |
    |  0  |   X[k,0]*X[k,0]    |  0.306   |
    |  0  | X[k-1,0]*X[k-3,0]  |  2.000   |
    |  0  |  X[k-2,0]*X[k,1]   |  1.528   |
    """
    xp, _, device_ = get_namespace_and_device(y)
    y = check_array(
        y,
        dtype=supported_float_dtypes(xp, device_),
        ensure_2d=False,
        ensure_all_finite="allow-nan",
    )
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    n_samples, n_outputs = y.shape

    if X is None:
        xy_hstack = y
        n_features = 0
    else:
        X = move_to(X, xp=xp, device=device_)
        X = check_array(
            X,
            dtype=y.dtype,
            ensure_2d=True,
            ensure_all_finite="allow-nan",
            ensure_min_features=0,
        )
        check_consistent_length(X, y)
        xy_hstack = xp.concat((X, y), axis=1)
        n_features = X.shape[1]

    if isinstance(n_terms_to_select, Integral):
        n_terms_to_select = np.full(n_outputs, n_terms_to_select, dtype=int)
    else:
        n_terms_to_select = column_or_1d(n_terms_to_select, dtype=int, warn=True)
        if len(n_terms_to_select) != n_outputs:
            raise ValueError(
                "The length of `n_terms_to_select` should be equal to "
                f"the number of outputs, {n_outputs}, but got "
                f"{len(n_terms_to_select)}."
            )
    session_sizes_cumsum = _validate_session_sizes(session_sizes, n_samples)

    # Session-local warmup bookkeeping. A session shorter than max_delay is
    # entirely warmup and retains no sample; it is reported (see
    # session_report_) instead of being dropped silently. The raw-data mask is
    # refined below with the mask actually applied by each selection path, so
    # rows dropped because a missing value propagated through delayed terms are
    # accounted for as well.
    X_np = (
        move_to(X, xp=np, device="cpu") if X is not None else np.empty((n_samples, 0))
    )
    y_np = move_to(y, xp=np, device="cpu")
    base_summary = _session_warmup_summary(
        np.ascontiguousarray(X_np, dtype=float),
        np.ascontiguousarray(y_np, dtype=float),
        session_sizes_cumsum,
        max_delay,
    )
    warmup_mask_np = base_summary["warmup_mask"]
    raw_non_finite_mask = ~(
        np.all(np.isfinite(X_np), axis=1) & np.all(np.isfinite(y_np), axis=1)
    )

    if include_zero_delay is None:
        _include_zero_delay = [True] * n_features + [False] * n_outputs
    else:
        _include_zero_delay = include_zero_delay + [False] * n_outputs

    time_shift_ids_all = make_time_shift_ids(
        n_features=xy_hstack.shape[1],
        max_delay=max_delay,
        include_zero_delay=_include_zero_delay,
    )

    if static_indices is None:
        static_indices = []

    time_shift_ids_all = np.delete(
        time_shift_ids_all,
        (
            np.isin(time_shift_ids_all[:, 0], static_indices)
            & (time_shift_ids_all[:, 1] > 0)
        ),
        0,
    )
    poly_ids_all = make_poly_ids(
        time_shift_ids_all.shape[0],
        poly_degree,
        max_poly=max_candidates,
        random_state=random_state,
    )

    if lazy:
        # Exclude every warmup row (including the first session's warmup and
        # the whole rows of sessions shorter than max_delay) and every
        # non-finite row. The mask is session-local, so a short session cannot
        # shift the warmup region of the next session. Generated features pad
        # delayed values with 0.0 instead of NaN, so missing values do not
        # propagate beyond the raw non-finite rows.
        valid_mask_np = ~warmup_mask_np & ~raw_non_finite_mask
        sample_mask = xp.asarray(valid_mask_np, device=device_)
        # LazyFastCan validates its inputs before applying sample_mask, so the
        # non-finite rows must be removed beforehand; the feature generator is
        # then bound to the compressed record and sees all its rows.
        xy_hstack_masked = xy_hstack[sample_mask]
        y_masked_lazy = y[sample_mask]
        gen_terms = partial(
            _gen_poly_time_shift_terms,
            time_shift_ids=time_shift_ids_all,
            poly_ids=poly_ids_all,
            mode="constant",
            constant_values=0.0,
            batch_size=batch_size,
        )
        selected_poly_ids = []
        for i in range(n_outputs):
            csf = LazyFastCan(
                n_features_to_select=n_terms_to_select[i],
                feature_generator=gen_terms,
            ).fit(xy_hstack_masked, y_masked_lazy[:, i])
            selected_ids = csf.indices_
            selected_ids = xp.sort(selected_ids)
            selected_ids = move_to(selected_ids, xp=np, device="cpu")
            selected_poly_ids.append(poly_ids_all[selected_ids])
        session_summary = _summarize_session_masks(
            warmup_mask_np,
            valid_mask_np,
            session_sizes_cumsum,
            non_finite_mask=raw_non_finite_mask,
        )
    else:
        poly_terms = _prepare_poly_terms(
            xy_hstack,
            time_shift_ids_all,
            poly_ids_all,
            session_sizes_cumsum,
            max_delay,
            xp=xp,
        )
        # Remove missing values. This is the authoritative valid-row mask of
        # the eager path: besides warmup and raw non-finite rows, it also
        # drops finite rows whose delayed terms inherit a missing value.
        eager_valid_mask = mask_missing_values(poly_terms, y, return_mask=True)
        eager_valid_mask = move_to(eager_valid_mask, xp=np, device="cpu")
        poly_terms_masked, y_masked = mask_missing_values(poly_terms, y)
        poly_terms_masked = move_to(poly_terms_masked, xp=np, device="cpu")
        y_masked = move_to(y_masked, xp=np, device="cpu")
        session_summary = _summarize_session_masks(
            warmup_mask_np,
            eager_valid_mask,
            session_sizes_cumsum,
            non_finite_mask=raw_non_finite_mask,
        )

        selected_poly_ids = []
        for i in range(n_outputs):
            csf = FastCan(
                n_terms_to_select[i],
                **params,
            ).fit(poly_terms_masked, y_masked[:, i])
            if refine_drop is not None:
                indices, _ = refine(
                    csf,
                    drop=refine_drop,
                    max_iter=refine_max_iter,
                    verbose=refine_verbose,
                )
                support = np.zeros(shape=poly_ids_all.shape[0], dtype=bool)
                support[indices] = True
            else:
                support = csf.get_support()
            selected_poly_ids.append(poly_ids_all[support])

    if max_delay > 0 and np.any(session_summary["n_valid_samples"] == 0):
        empty_sessions = np.flatnonzero(
            session_summary["n_valid_samples"] == 0
        ).tolist()
        warnings.warn(
            "The following sessions contain no valid sample for candidate "
            f"selection and are excluded in full: {empty_sessions}. Their "
            f"size is not larger than the maximum delay {max_delay}, so all "
            "their rows are used as warmup values.",
            UserWarning,
        )

    selected_poly_ids = np.vstack(selected_poly_ids)

    time_shift_ids = time_shift_ids_all[
        np.unique(selected_poly_ids[selected_poly_ids != -1]), :
    ]
    poly_ids = (
        rankdata(
            np.r_[[[-1] * poly_degree], selected_poly_ids],
            method="dense",
        ).reshape(-1, poly_degree)[1:]
        - 2
    )

    output_ids = [i for i in range(n_outputs) for _ in range(n_terms_to_select[i])]
    feat_ids, delay_ids = tp2fd(time_shift_ids, poly_ids)

    narx = NARX(
        feat_ids=feat_ids,
        delay_ids=delay_ids,
        output_ids=output_ids,
        fit_intercept=fit_intercept,
    )
    # Candidate-filter bookkeeping, available before fit() as well.
    narx.session_report_ = {
        "session_sizes": session_summary["session_sizes"],
        "n_valid_samples": session_summary["n_valid_samples"],
        "n_excluded_samples": session_summary["n_excluded_samples"],
        "exclusion_reasons": session_summary["exclusion_reasons"],
    }
    if return_report:
        return narx, narx.session_report_
    return narx


def _gen_poly_time_shift_terms(
    X, time_shift_ids, poly_ids, skip_indices=None, batch_size=16, **kwargs
):
    """Generate polynomial time shift terms."""
    xp, _, device_ = get_namespace_and_device(X)
    n_samples = X.shape[0]
    n_features = poly_ids.shape[0]
    valid_indices = _skip2valid(skip_indices, n_features, xp=xp, device=device_)

    n_valid = valid_indices.shape[0]
    for i in range(0, n_valid, batch_size):
        batch_idx = valid_indices[i : i + batch_size]
        batch_features = xp.ones(
            (n_samples, len(batch_idx)), dtype=X.dtype, device=device_
        )
        for j, feat_id in enumerate(batch_idx):
            for var_id in poly_ids[feat_id]:
                if var_id != -1:
                    time_shift_vars = _make_a_time_shift_feature(
                        X, time_shift_ids[var_id], **kwargs
                    )
                    batch_features[:, j] *= time_shift_vars
        yield batch_idx, batch_features
