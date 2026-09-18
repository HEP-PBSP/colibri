"""
colibri.analytic_fit.py

For a linear model, this module allows for an analytic Bayesian fit of the
model.

"""

import time

import jax
import jax.numpy as jnp
import jax.numpy.linalg as jla
import jax.lax.linalg as jlinalg
import numpy as np
import scipy.special as special
import scipy.linalg as sla

from validphys import convolution
from validphys.fkparser import load_fktable

from colibri.core import AnalyticFit
from colibri.export_results import write_replicas, export_bayes_results
from colibri.checks import check_pdf_model_is_linear
from colibri.utils import compute_determinants_of_principal_minors
from colibri.theory_predictions import fktable_xgrid_indices

import logging

log = logging.getLogger(__name__)

# Size of the evolution-basis flavour space that pdf_model.grid_values_func
# returns PDF values on (the "N_fl" in the (N_fl, Nx) shape documented in
# colibri.pdf_model.PDFModel.grid_values_func), same basis as
# colibri.constants.FLAVOUR_TO_ID_MAPPING / colibri.compute_svd.N_FLAVOURS.
N_FLAVOURS = len(convolution.FK_FLAVOURS)


def analytic_evidence_uniform_prior(sol_covmat, sol_mean, max_logl, a_vec, b_vec):
    """
    Compute the log of the evidence for Gaussian likelihood and uniform prior.
    The implementation is based on the following paper: https://arxiv.org/pdf/2301.13783
    and consists in a small improvement of the Laplace approximation.

    Parameters
    ----------
    sol_covmat: jnp.ndarray
        Covariance matrix of the posterior (X^T Sigma^-1 X)^-1.

    sol_mean: jnp.ndarray
        Posterior mean vector.

    a_vec: np.ndarray
        Lower bounds of the Uniform prior.

    b_vec: np.ndarray
        Upper bounds of the Uniform prior.

    Returns
    -------
    tuple[float, float]
        The log evidence and the log Occam factor.
    """

    # Take into account change of variables of type (x - mu) -> x
    b_vec -= sol_mean
    a_vec -= sol_mean

    determinants = compute_determinants_of_principal_minors(sol_covmat)

    sqrt_det_ratios = np.sqrt(determinants[:-1] / determinants[1:])

    erf_arg_a = a_vec / np.sqrt(2) * sqrt_det_ratios
    erf_arg_b = b_vec / np.sqrt(2) * sqrt_det_ratios

    erf_a = special.erf(erf_arg_a)
    erf_b = special.erf(erf_arg_b)

    log_erf_terms = np.log(0.5 * (erf_b - erf_a)).sum()

    occam_factor_num = np.sqrt(jla.det(sol_covmat))
    occam_factor_denom = np.prod((b_vec - a_vec))

    log_occam_factor = np.log(occam_factor_num / occam_factor_denom)

    log_evidence = (
        max_logl
        + sol_covmat.shape[0] / 2 * np.log(2 * np.pi)
        + log_occam_factor
        + log_erf_terms
    )

    return log_evidence, log_occam_factor


def _combined_dis_fk_matrix(data, FIT_XGRID):
    """
    Builds the raw (un-whitened) combined DIS FK matrix entering ``data``,
    padded onto the full (N_FLAVOURS, len(FIT_XGRID)) basis -- i.e. exactly
    the object diagnosed by ``colibri.compute_svd.compute_svd`` (full-basis
    version). Used here to regularise the *forward operator* itself via
    TSVD before it is composed with the (much lower-dimensional) PDF
    parametrisation.

    Hadronic FK tables are skipped (with a warning): they enter
    predictions bilinearly in the PDF and are not part of a linear forward
    operator, so an analytic fit could not have included them in the first
    place (``check_pdf_model_is_linear`` would already have failed).

    Parameters
    ----------
    data : validphys.core.DataGroupSpec
    FIT_XGRID : array-like

    Returns
    -------
    np.ndarray
        Array of shape (Ndat, N_FLAVOURS * len(FIT_XGRID)), datasets
        stacked in the same order as ``data.datasets``.
    """
    blocks = []
    skipped = []

    for ds in data.datasets:
        for i, fkspec in enumerate(ds.fkspecs):
            fk = load_fktable(fkspec).with_cuts(ds.cuts)

            if fk.hadronic:
                skipped.append(f"{ds.name}_fk{i}")
                continue

            fk_arr = np.asarray(fk.get_np_fktable())  # (Ndat, n_present_fl, Nx_local)
            n_dat = fk_arr.shape[0]

            lumi_indices = np.asarray(fk.luminosity_mapping)
            x_indices = np.asarray(fktable_xgrid_indices(fk, FIT_XGRID))

            padded = np.zeros((n_dat, N_FLAVOURS, len(FIT_XGRID)))
            padded[:, lumi_indices[:, None], x_indices[None, :]] = fk_arr

            blocks.append(padded.reshape(n_dat, -1))

    if skipped:
        log.warning(
            f"_combined_dis_fk_matrix: skipped {len(skipped)} hadronic FK "
            f"table(s), not usable in a linear/analytic fit: {skipped}"
        )

    if not blocks:
        raise ValueError("No DIS FK tables found in `data` -- cannot build the FK operator.")

    return np.concatenate(blocks, axis=0)


def _build_whitened_linear_system(
    central_covmat_index,
    forward_map,
    analytic_settings,
    fast_kernel_arrays,
    data,
    FIT_XGRID,
):
    """
    Builds the whitened linear system X_tilde w ~ Y_tilde entering the
    analytic fit's chi2, including the FK-level TSVD regularisation
    (``analytic_settings["tsvd_n_components"]``) if set. Factored out so it
    can be shared between ``analytic_fit`` (the actual solve) and
    ``compute_lcurve`` (an L-curve diagnostic over a grid of L2/Tikhonov
    regularisation strengths, applied on top of whatever TSVD truncation is
    already set here).

    See ``analytic_fit``'s docstring for the meaning of T(w) = T(0) + X w,
    Y = D - T(0), and of the FK-level TSVD regularisation.

    Returns
    -------
    X_tilde : jnp.ndarray, shape (Ndat, n_params)
    Y_tilde : jnp.ndarray, shape (Ndat,)
    parameters : list of str
        forward_map.param_names.
    B : np.ndarray, shape (N_FLAVOURS * len(FIT_XGRID), n_params)
        Basis-to-grid matrix: column i is the flattened (N_fl, Nx) PDF
        grid produced by the model for unit parameter vector e_i. Always
        computed (independently of whether TSVD is enabled), so that
        grid-space regularisation penalties (e.g. in ``compute_lcurve``)
        can be built via L_eff = L_grid @ B regardless of the TSVD
        setting.
    """
    # Ensure that the PDF model is linear before running the fit.
    log.info("Checking that the PDF model is linear...")
    check_pdf_model_is_linear(forward_map, fast_kernel_arrays)

    parameters = forward_map.param_names
    n_params = len(parameters)

    # Precompute predictions (and PDF grids) for the basis of the model
    bases = jnp.identity(n_params)
    pred_pdf_pairs = [forward_map(fast_kernel_arrays, basis) for basis in bases]
    predictions = jnp.array([pred for pred, _ in pred_pdf_pairs])
    pdf_grids = jnp.array([grid for _, grid in pred_pdf_pairs])  # (n_params, N_fl, Nx)

    intercept, intercept_pdf = forward_map(fast_kernel_arrays, jnp.zeros(n_params))

    n_fl, n_x = pdf_grids.shape[1], pdf_grids.shape[2]
    if n_x != len(FIT_XGRID):
        raise ValueError(
            f"pdf grid has {n_x} x-points but FIT_XGRID has {len(FIT_XGRID)}; "
            "these must match to build grid-space regularisation."
        )
    if n_fl != N_FLAVOURS:
        raise ValueError(
            f"pdf grid has {n_fl} flavours but the FK operator basis "
            f"expects N_FLAVOURS={N_FLAVOURS}; check pdf_model.grid_values_func."
        )
    n_grid = n_fl * n_x

    # B: (n_grid, n_params) basis-to-grid matrix; columns are pdf(e_i),
    # flattened in the same (N_fl, Nx) convention as the FK operator.
    # Computed unconditionally (cheap -- pdf_grids is already in memory)
    # since grid-space regularisation (compute_lcurve) needs it whether or
    # not TSVD is also enabled.
    B = np.asarray(pdf_grids).reshape(n_params, n_grid).T

    central_values = central_covmat_index.central_values
    covmat = central_covmat_index.covmat

    tsvd_n_components = analytic_settings.get("tsvd_n_components", None)

    if tsvd_n_components is None:
        # Default: no regularisation of the forward operator.
        Y = central_values - intercept
        X = predictions.T - intercept[:, None]
    else:
        intercept_grid_flat = np.asarray(intercept_pdf).reshape(n_grid)

        fk_flat = _combined_dis_fk_matrix(data, FIT_XGRID)  # (Ndat, n_grid)

        max_components = min(fk_flat.shape)
        if tsvd_n_components > max_components:
            raise ValueError(
                f"tsvd_n_components={tsvd_n_components} cannot exceed "
                f"min(Ndat, N_FLAVOURS * len(FIT_XGRID))={max_components}."
            )

        log.warning(
            f"Applying TSVD regularisation to the FK operator: keeping the "
            f"top {tsvd_n_components} out of {max_components} singular "
            f"values/vectors (FK operator shape {fk_flat.shape})."
        )

        U, S, Vt = np.linalg.svd(fk_flat, full_matrices=False)
        U_k = U[:, :tsvd_n_components]
        S_k = S[:tsvd_n_components]
        Vt_k = Vt[:tsvd_n_components, :]

        log.info(
            f"Discarded singular values range from {S[tsvd_n_components]:.3e} "
            f"down to {S[-1]:.3e}; kept singular values range from "
            f"{S_k[0]:.3e} down to {S_k[-1]:.3e}."
        )

        # FK_trunc @ v = U_k @ (S_k * (V_k^T @ v)) for any vector/matrix v,
        # without ever forming the full (Ndat, n_grid) FK_trunc explicitly.
        def apply_fk_trunc(v):
            return U_k @ (S_k[:, None] * (Vt_k @ v)) if v.ndim > 1 else U_k @ (
                S_k * (Vt_k @ v)
            )

        X_trunc_np = apply_fk_trunc(B)  # (Ndat, n_params)
        intercept_trunc_np = apply_fk_trunc(intercept_grid_flat)  # (Ndat,)

        # Sanity check: with no truncation (k == max_components) this should
        # reproduce the un-regularised intercept/X from forward_map to high
        # precision -- logged here to help validate the FK/pdf padding and
        # dataset-ordering conventions line up.
        if tsvd_n_components == max_components:
            diff = np.max(np.abs(intercept_trunc_np - np.asarray(intercept)))
            log.info(
                f"TSVD self-consistency check (k=max_components): max "
                f"|intercept_trunc - intercept| = {diff:.3e} (should be ~0)."
            )

        X = jnp.asarray(X_trunc_np)
        intercept_reg = jnp.asarray(intercept_trunc_np)
        Y = central_values - intercept_reg

    # Cholesky factorization: S = L L^T
    # upper False means that we want the lower triangular matrix L
    L = jla.cholesky(covmat, upper=False)

    # Whiten the problem: Y' = L^-1 Y, X' = L^-1 X
    Y_tilde = jlinalg.triangular_solve(L, Y, left_side=True, lower=True)
    X_tilde = jlinalg.triangular_solve(L, X, left_side=True, lower=True)

    return X_tilde, Y_tilde, parameters, B, n_fl, n_x


def _second_order_roughening_matrix(FIT_XGRID, n_fl):
    """
    Builds the block-diagonal (per flavour) second-order roughening matrix
    L_grid acting on a flattened (n_fl, len(FIT_XGRID)) PDF grid, i.e. the
    spacing-corrected generalisation of the constant-spacing L2 stencil of
    eq. (4.28) in Aster, Borchers & Thurber, "Parameter Estimation and
    Inverse Problems" (2013).

    For a non-uniform grid x_0, ..., x_{Nx-1}, the interior rows use the
    standard three-point finite-difference approximation to f''(x_i):

        f''(x_i) ~ 2 * [ f_{i-1} / (h_i^- (h_i^- + h_i^+))
                          - f_i   / (h_i^- h_i^+)
                          + f_{i+1} / (h_i^+ (h_i^- + h_i^+)) ]

    where h_i^- = x_i - x_{i-1} and h_i^+ = x_{i+1} - x_i. This reduces
    exactly to the constant-spacing [1, -2, 1] stencil of eq. (4.28) when
    the grid is uniform (h_i^- = h_i^+ = h for all i, up to the overall
    1/h^2 normalisation).

    No cross-flavour differencing is applied (flavours are not neighbours
    in x), so the full operator is block-diagonal across the n_fl
    flavours, each block being the (Nx - 2, Nx) stencil above.

    Parameters
    ----------
    FIT_XGRID : array-like
        The (possibly non-uniform) x-grid, length Nx.

    n_fl : int
        Number of flavours (grid rows); N_FLAVOURS in the un-truncated
        case, or len(flavour_indices) if restricting to a subset.

    Returns
    -------
    np.ndarray of shape (n_fl * (Nx - 2), n_fl * Nx)
    """
    x = np.asarray(FIT_XGRID, dtype=float)
    n_x = len(x)

    if n_x < 3:
        raise ValueError("FIT_XGRID must have at least 3 points for a second-order penalty.")

    L1 = np.zeros((n_x - 2, n_x))
    for i in range(1, n_x - 1):
        h_minus = x[i] - x[i - 1]
        h_plus = x[i + 1] - x[i]
        L1[i - 1, i - 1] = 2.0 / (h_minus * (h_minus + h_plus))
        L1[i - 1, i] = -2.0 / (h_minus * h_plus)
        L1[i - 1, i + 1] = 2.0 / (h_plus * (h_minus + h_plus))

    # Block-diagonal across flavours; ordering matches the (n_fl, Nx)
    # row-major flattening used everywhere else (B, the FK operator, etc.)
    return np.kron(np.eye(n_fl), L1)


def compute_lcurve(
    central_covmat_index,
    forward_map,
    analytic_settings,
    fast_kernel_arrays,
    data,
    FIT_XGRID,
    output_path,
    lcurve_settings=None,
):
    """
    Computes the L-curve for second-order Tikhonov regularisation (eq. 4.25
    and 4.28 in Aster, Borchers & Thurber), applied *on top* of whatever
    FK-level TSVD truncation is set via
    ``analytic_settings["tsvd_n_components"]`` (if any). This is purely a
    diagnostic to help choose a regularisation strength lambda by eye
    (looking for the "corner" of the L-curve) -- it does not change the
    actual fit result computed by ``analytic_fit``.

    Where the roughness penalty lives
    ----------------------------------
    The fit still solves for the n_params (e.g. 43) basis-function
    weights w -- that does not change. However "smoothness" is only a
    meaningful notion on the reconstructed (N_FLAVOURS, Nx) PDF grid
    (consecutive weights are not, in general, neighbours; consecutive
    x-grid points are). So the roughening matrix L_grid (see
    ``_second_order_roughening_matrix``) is built directly on that grid,
    exactly like the FK-level TSVD operator is, and then pulled back onto
    the weights via the same basis-to-grid matrix B used for TSVD:

        L_eff = L_grid @ B,   shape (N_FLAVOURS * (Nx - 2), n_params)

    so that the penalised quantity is the seminorm of the *reconstructed
    grid*, ||L_grid @ (B w)||_2 = ||L_eff @ w||_2, restricted to the part
    of the grid spanned by the fit's own parametrisation (the fixed
    intercept T(0)/pdf(0) is not part of what is optimised over, so it is
    excluded from the penalty).

    For each lambda in a grid, the regularised solution solves the
    (n_params x n_params) normal equations directly (cheap, since
    n_params is small -- no need for the GSVD machinery of section 4.4,
    which exists purely for computational efficiency on much larger
    problems than this one):

        (X_tilde^T X_tilde + lambda^2 L_eff^T L_eff) w_lambda = X_tilde^T Y_tilde

    and this function records:

    - residual_norm(lambda) = ||Y_tilde - X_tilde w_lambda||
    - seminorm(lambda) = ||L_eff w_lambda||

    Writes ``<output_path>/lcurve/lcurve.txt`` with columns
    "lambda,residual_norm,seminorm".

    Parameters
    ----------
    central_covmat_index, forward_map, analytic_settings, fast_kernel_arrays,
    data, FIT_XGRID : see ``analytic_fit``.

    output_path : pathlib.Path
        Colibri output folder. Automatically provided by
        ``colibri.config.Environment``.

    lcurve_settings : dict, default is None
        Optional dict with keys:

        - "n_lambda" (int, default 100): number of lambda values.
        - "lambda_min" (float, default S[-1] * 1e-3): smallest lambda.
        - "lambda_max" (float, default S[0] * 1e3): largest lambda.

        where S are the singular values of X_tilde. lambda values are
        log-spaced between lambda_min and lambda_max (see
        ``compute_lcurve``'s module-level discussion for why a grid of
        lambda is needed at all: the L-curve is the parametric curve
        traced out by lambda, not a single computable point). Can be set
        directly in the runcard.

    Returns
    -------
    tuple of np.ndarray
        (lambdas, residual_norms, seminorms), the same arrays written to
        the CSV.
    """
    X_tilde, Y_tilde, _, B, n_fl, n_x = _build_whitened_linear_system(
        central_covmat_index,
        forward_map,
        analytic_settings,
        fast_kernel_arrays,
        data,
        FIT_XGRID,
    )

    X_tilde = np.asarray(X_tilde)
    Y_tilde = np.asarray(Y_tilde)
    n_params = X_tilde.shape[1]

    L_grid = _second_order_roughening_matrix(FIT_XGRID, n_fl)
    L_eff = L_grid @ B  # (n_fl * (n_x - 2), n_params)

    log.info(
        f"Second-order roughening matrix built on the ({n_fl}, {n_x}) grid, "
        f"pulled back to L_eff with shape {L_eff.shape} via B."
    )

    XtX = X_tilde.T @ X_tilde  # (n_params, n_params), fixed across lambda
    XtY = X_tilde.T @ Y_tilde  # (n_params,), fixed across lambda
    LtL = L_eff.T @ L_eff  # (n_params, n_params), fixed across lambda

    # lambda grid defaults: the natural scale for lambda is set by the
    # *generalized* singular values gamma_i (section 4.4 of Aster, Borchers
    # & Thurber), i.e. the square roots of the generalized eigenvalues of
    # the pencil (X_tilde^T X_tilde, L_eff^T L_eff): X^TX v = gamma^2 L^TL v.
    # Using only X_tilde's own singular values (as an earlier version of
    # this function did) ignores L_eff's scale entirely -- and L_eff, built
    # from second-derivative finite differences 1/h^2 on a possibly very
    # non-uniform FIT_XGRID (tiny h near small x), can have a wildly
    # different magnitude than X_tilde. Getting this wrong means the swept
    # lambda range can land entirely in one asymptotic tail of the true
    # L-curve, missing the corner (and both flat segments) completely --
    # producing a curve that looks like a single steep drop in the wrong
    # place rather than a proper "L".
    try:
        gen_eigvals = sla.eigh(XtX, LtL, eigvals_only=True)
        gen_eigvals = np.clip(gen_eigvals, 1e-300, None)
        gammas = np.sqrt(gen_eigvals)
        default_lambda_min = gammas.min() * 1e-2
        default_lambda_max = gammas.max() * 1e2
        log.info(
            f"Generalized singular values (gamma_i) of (X_tilde, L_eff) "
            f"range from {gammas.min():.3e} to {gammas.max():.3e}."
        )
    except Exception as exc:
        # Falls back to the old X_tilde-only heuristic if L_eff is rank
        # deficient (LtL not positive definite) or the generalized
        # eigenproblem otherwise fails to solve.
        log.warning(
            f"Generalized eigenvalue problem for the lambda range failed "
            f"({exc}); falling back to a heuristic based on X_tilde's own "
            f"singular values only. Consider setting lambda_min/lambda_max "
            f"manually via lcurve_settings if the resulting L-curve looks off."
        )
        S = np.linalg.svd(X_tilde, compute_uv=False)
        default_lambda_min = S[-1] * 1e-3
        default_lambda_max = S[0] * 1e3

    lcurve_settings = lcurve_settings or {}
    n_lambda = lcurve_settings.get("n_lambda", 100)
    lambda_min = lcurve_settings.get("lambda_min", default_lambda_min)
    lambda_max = lcurve_settings.get("lambda_max", default_lambda_max)

    log.info(
        f"Computing L-curve: {n_lambda} lambda values log-spaced between "
        f"{lambda_min:.3e} and {lambda_max:.3e}."
    )

    lambdas = np.logspace(np.log10(lambda_min), np.log10(lambda_max), n_lambda)

    residual_norms = np.empty(n_lambda)
    seminorms = np.empty(n_lambda)

    for j, lam in enumerate(lambdas):
        w_lambda = np.linalg.solve(XtX + lam**2 * LtL, XtY)
        residual_norms[j] = np.linalg.norm(Y_tilde - X_tilde @ w_lambda)
        seminorms[j] = np.linalg.norm(L_eff @ w_lambda)

    lcurve_folder = output_path / "lcurve"
    lcurve_folder.mkdir(exist_ok=True)
    lcurve_path = lcurve_folder / "lcurve.txt"
    np.savetxt(
        lcurve_path,
        np.column_stack([lambdas, residual_norms, seminorms]),
        header="lambda,residual_norm,seminorm",
        delimiter=",",
        comments="",
    )
    log.info(f"L-curve ({n_lambda} lambda values) saved to {lcurve_path}")

    return lambdas, residual_norms, seminorms


def analytic_fit(
    central_covmat_index,
    forward_map,
    analytic_settings,
    prior_settings,
    fast_kernel_arrays,
    data,
    FIT_XGRID,
):
    """
    Analytic fits, for any *linear* PDF model.

    The assumption is that the model is linear with an intercept:
    T(w) = T(0) + X w.
    The linear problem to solve is through minimisation of the chi2:
    chi2 = (D - (T(0) + X w))^T Sigma^-1 (D - (T(0) + X w)) = (Y - X w)^T Sigma^-1 (Y - X w)
    with Y = D - T(0).

    Optional TSVD regularisation (``analytic_settings["tsvd_n_components"]``)
    -------------------------------------------------------------------------
    T(w) itself factorises as T(w) = FK @ pdf(w), where FK is the raw
    (Ndat, N_FLAVOURS * Nx) FK operator (see ``_combined_dis_fk_matrix``,
    the same object diagnosed by ``colibri.compute_svd``) and pdf(w) is the
    (N_FLAVOURS * Nx,)-flattened PDF grid produced by the model's
    ``grid_values_func``. Since the fit parametrisation typically has far
    fewer parameters than N_FLAVOURS * Nx (e.g. 43 vs. 700), truncating the
    *parameter-space* design matrix X (shape (Ndat, n_params)) can only ever
    keep at most n_params components -- there is nothing to truncate there.

    Instead, when ``tsvd_n_components`` is set, this function truncates the
    FK operator itself: FK = U S V^T -> FK_trunc = U_k S_k V_k^T (keeping
    the k=tsvd_n_components largest singular values/vectors), and then
    composes the *truncated* operator with the model's basis:
    X_trunc = FK_trunc @ B, intercept_trunc = FK_trunc @ pdf(0), where B is
    the (N_FLAVOURS * Nx, n_params) matrix whose columns are pdf(e_i) for
    each unit parameter vector e_i (obtained directly from the ``pdf``
    output of ``forward_map``). The subsequent whitening + exact QR solve
    are unchanged and operate on this regularised X_trunc/intercept_trunc,
    so the posterior covariance remains full-rank (n_params x n_params) and
    the evidence formulas below stay valid.

    See ``compute_lcurve`` for how to choose ``analytic_settings["l2_lambda"]``
    via the L-curve criterion before setting it here.

    Optional L2/Tikhonov (second-order) regularisation (``analytic_settings["l2_lambda"]``)
    -------------------------------------------------------------------------------------
    Applied on top of whatever TSVD truncation is set above (independent
    setting -- can be used alone, together with TSVD, or not at all).
    Penalises the curvature (second derivative, spacing-corrected for a
    non-uniform FIT_XGRID) of the reconstructed PDF grid B w, pulled back
    onto the n_params weights exactly as TSVD pulls the FK operator back
    onto them: L_eff = L_grid @ B (see ``_second_order_roughening_matrix``
    and ``compute_lcurve``, which computes the L-curve diagnostic for this
    same L_eff without applying it to the fit). When
    ``analytic_settings["l2_lambda"]`` is set to a float, the exact QR
    solve below is replaced by the regularised normal equations

        (X_tilde^T X_tilde + l2_lambda^2 L_eff^T L_eff) sol_mean = X_tilde^T Y_tilde
        sol_covmat = (X_tilde^T X_tilde + l2_lambda^2 L_eff^T L_eff)^-1

    which is exactly the Bayesian MAP/posterior-covariance solution under
    an additional Gaussian prior on w with precision l2_lambda^2 L_eff^T L_eff
    (mean zero) -- i.e. still a well-defined Gaussian posterior, so the
    evidence formulas below remain valid. When ``l2_lambda`` is None
    (default), this reduces exactly to the original unregularised QR
    solve.

    Parameters
    ----------
    central_covmat_index: commondata_utils.CentralCovmatIndex
        dataclass containing central values and covariance matrix.

    forward_map: @jax.jit CompiledFunction
        Forward map function for the fit.

    analytic_settings: dict
        Settings for the analytic fit. May contain "tsvd_n_components"
        (int or None) to enable FK-level TSVD regularisation, and
        "l2_lambda" (float or None) to enable second-order Tikhonov
        regularisation on top of it.

    prior_settings: PriorSettings
        Settings for the prior.

    fast_kernel_arrays: tuple
        Tuple containing the fast kernel arrays.

    data: validphys.core.DataGroupSpec
        The data entering the fit. Only used when ``tsvd_n_components`` is
        set, to rebuild the raw FK operator for truncation.

    FIT_XGRID: np.ndarray
        Common fit x-grid. Used when ``tsvd_n_components`` and/or
        ``l2_lambda`` are set.
    """
    log.warning("The prior is assumed to be flat in the parameters.")
    log.warning(
        "Assuming that the prior is wide enough to fully cover the gaussian likelihood."
    )

    covmat = central_covmat_index.covmat

    X_tilde, Y_tilde, parameters, B, n_fl, n_x = _build_whitened_linear_system(
        central_covmat_index,
        forward_map,
        analytic_settings,
        fast_kernel_arrays,
        data,
        FIT_XGRID,
    )

    l2_lambda = analytic_settings.get("l2_lambda", None)

    t0 = time.time()

    if l2_lambda is None:
        if jnp.any(jla.eigh(X_tilde.T @ X_tilde)[0] <= 0.0):
            raise ValueError(
                "The obtained covariance matrix for the analytic solution is not positive definite."
            )

        # Compute QR decomposition of X_tilde for numerical stability in the inversion
        Q, R = jla.qr(X_tilde)

        # NOTE: R is upper triangular in QR decomposition, so we need to set lower=False
        sol_mean = jlinalg.triangular_solve(R, Q.T @ Y_tilde, left_side=True, lower=False)

        I_R = jnp.eye(R.shape[0])
        R_inv = jlinalg.triangular_solve(R, I_R, left_side=True, lower=False)
        sol_covmat = R_inv @ R_inv.T
    else:
        log.warning(
            f"Applying L2/second-order Tikhonov regularisation with "
            f"l2_lambda = {l2_lambda:.6e} (chosen e.g. via the L-curve "
            f"corner from compute_lcurve)."
        )

        L_grid = _second_order_roughening_matrix(FIT_XGRID, n_fl)
        L_eff = jnp.asarray(L_grid @ B)  # (n_fl * (n_x - 2), n_params)

        XtX = X_tilde.T @ X_tilde
        LtL = L_eff.T @ L_eff
        A = XtX + l2_lambda**2 * LtL

        if jnp.any(jla.eigh(A)[0] <= 0.0):
            raise ValueError(
                "The regularised (X_tilde^T X_tilde + l2_lambda^2 L_eff^T L_eff) "
                "matrix is not positive definite."
            )

        sol_covmat = jla.inv(A)
        sol_mean = sol_covmat @ (X_tilde.T @ Y_tilde)

    key = jax.random.PRNGKey(analytic_settings["sampling_seed"])

    # full samples with no cuts from the prior bounds
    full_samples = jax.random.multivariate_normal(
        key,
        sol_mean,
        sol_covmat,
        shape=(analytic_settings["full_sample_size"],),
    )

    # Compute the evidence
    # This is the log of the evidence, which is the log of the integral of the likelihood
    # over the prior. The prior is uniform with width prior_width.
    log.info("Computing the evidence...")

    if prior_settings.prior_distribution == "n_sigma_prior":
        nsigma = prior_settings.prior_distribution_specs["n_sigma_value"]

        log.info(f"Using +- {nsigma} sigma of covmat")
        diags = np.sqrt(np.diag(sol_covmat))

        prior_lower = sol_mean - nsigma * diags
        prior_upper = sol_mean + nsigma * diags

    elif prior_settings.prior_distribution == "custom_uniform_parameter_prior":
        log.info("Using custom uniform prior")
        prior_lower = jnp.array(prior_settings.prior_distribution_specs["lower_bounds"])
        prior_upper = jnp.array(prior_settings.prior_distribution_specs["upper_bounds"])

    elif prior_settings.prior_distribution == "min_max_prior":
        log.info("Using min-max prior")
        prior_lower = full_samples.min(axis=0)
        prior_upper = full_samples.max(axis=0)

    else:
        # Extract lower and upper bounds of the prior
        prior_lower = prior_settings.prior_distribution_specs["min_val"] * jnp.ones(
            len(parameters)
        )
        prior_upper = prior_settings.prior_distribution_specs["max_val"] * jnp.ones(
            len(parameters)
        )

    prior_width = prior_upper - prior_lower

    # Check that the prior is wide enough
    if jnp.any(full_samples < prior_lower) or jnp.any(full_samples > prior_upper):
        log.error(
            "The prior is not wide enough to cover the posterior samples. Increase the prior width."
        )

    log.warning(f"Discarding samples outside the prior bounds.")

    # discard samples outside the prior
    full_samples = full_samples[
        (full_samples > prior_lower).all(axis=1)
        & (full_samples < prior_upper).all(axis=1)
    ]

    gaussian_integral = jnp.log(jnp.sqrt(jla.det(2 * jnp.pi * sol_covmat)))
    log_prior = jnp.log(1 / prior_width).sum()
    # Compute maximum log likelihood in the whitened basis
    min_chi2 = (Y_tilde - X_tilde @ sol_mean).T @ (Y_tilde - X_tilde @ sol_mean)
    # Compute the log likelihood
    max_logl = -0.5 * min_chi2

    logZ_laplace = gaussian_integral + max_logl + log_prior

    log.info(f"LogZ (Laplace approximation) = {logZ_laplace}")

    # computation of the evidence (analytic approximation)
    logZ_analytical, log_occam_factor = analytic_evidence_uniform_prior(
        sol_covmat, sol_mean, max_logl, prior_lower, prior_upper
    )

    log.info(f"LogZ (Analytic approximation) = {logZ_analytical}")
    log.info(f"Log Occam factor = {log_occam_factor}")
    log.info(f"Maximal log likelihood = {max_logl}")

    # Compute minimum chi2
    min_chi2 = -2 * max_logl
    log.info(f"Minimum chi2 = {min_chi2}")

    BIC = min_chi2 + sol_covmat.shape[0] * np.log(covmat.shape[0])
    AIC = min_chi2 + 2 * sol_covmat.shape[0]

    # Compute average chi2 (in whitened basis)
    diffs = Y_tilde[:, None] - X_tilde @ full_samples.T
    avg_chi2 = jnp.mean(jnp.sum(diffs**2, axis=0))

    log.info(f"Average chi2 = {avg_chi2}")

    # Compute the Bayesian complexity
    Cb = avg_chi2 - min_chi2
    log.info(f"Bayesian complexity = {Cb}")

    # Resample the posterior for PDF set
    samples = full_samples[: analytic_settings["n_posterior_samples"]]

    t1 = time.time()
    log.info("ANALYTIC SAMPLING RUNTIME: %f s" % (t1 - t0))

    return AnalyticFit(
        analytic_specs=analytic_settings,
        resampled_posterior=samples,
        param_names=parameters,
        full_posterior_samples=full_samples,
        bayesian_metrics={
            "bayes_complexity": Cb,
            "avg_chi2": avg_chi2,
            "min_chi2": min_chi2,
            "logZ_laplace": logZ_laplace,
            "logz": logZ_analytical,
            "log_occam_factor": log_occam_factor,
            "BIC": BIC,
            "AIC": AIC,
        },
    )


def run_analytic_fit(analytic_fit, output_path, pdf_model, Q0):
    """
    Export the results of an analytic fit.

    Parameters
    ----------
    analytic_fit: AnalyticFit
        The results of the analytic fit.
    output_path: pathlib.PosixPath
        Path to the output folder.
    pdf_model: pdf_model.PDFModel
        The PDF model used in the fit.
    Q0: float
        The scale at which to export the PDFs.
    """

    export_bayes_results(analytic_fit, output_path, "analytic_result")

    write_replicas(analytic_fit, output_path, pdf_model, Q0)