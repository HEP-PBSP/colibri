from colibri.export_results import write_replicas

# from colibri.export_results import export_hessian_results
import jax.numpy as jnp
from dataclasses import dataclass
import logging
import time
import jax
from colibri.gradient_descent import run_gradient_descent
from colibri.param_initialisation import pdf_initial_parameters

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class HessianFit:
    """
    Dataclass containing the results and specs of a Hessian fit.

    Attributes
    ----------
    hessian_specs: dict
        Dictionary containing the settings of the Hessian fit.
    min_chi2: jnp.ndarray
        Array containing the minimum chi-squared value.
    training_loss: jnp.ndarray
        Array containing the training loss values during the fit.
    optimized_parameters: jnp.ndarray
        Array containing the optimized parameters in the minimum of the chi2.
    hessian: jnp.ndarray
        Array containing the Hessian matrix at the minimum of the chi2.
    cov_params: jnp.ndarray
        Array containing the covariance matrix of the parameters.
        Computed as the inverse of the Hessian matrix.
    resampled_posterior: jnp.ndarray
        Array containing the samples of the parameters drawn from a multivariate
        normal distribution with mean at the optimized parameters and covariance
        given by cov_params.
    """

    hessian_specs: dict
    min_chi2: jnp.ndarray
    training_loss: jnp.ndarray
    optimized_parameters: jnp.ndarray
    hessian: jnp.ndarray
    cov_params: jnp.ndarray
    resampled_posterior: jnp.ndarray


def hessian_fit(
    pdf_model,
    log_likelihood,
    optimizer_provider,
    early_stopper,
    max_epochs,
    hessian_settings,
    param_initialiser_settings,
):

    log.info(f"Running fit with backend: {jax.lib.xla_bridge.get_backend().platform}")
    log.info("Starting Hessian fit...")

    # run_gradient_descent expects a data batch object, but we don't use it here
    def train_chi2(params, idx):
        return -2 * log_likelihood(params)

    def valid_chi2(params):
        return jnp.nan

    iter_init = hessian_settings["iter_init"]
    tolerance = hessian_settings["tolerance"]
    # Optional local-minimum checks
    grad_tol = hessian_settings.get("grad_tol", 1e-6)
    eig_eps = hessian_settings.get("min_hessian_eigval", 1e-12)
    require_local_min = hessian_settings.get("require_local_min", False)

    t0 = time.time()
    # Generate iter_init random initial parameters
    # run the fit and pick the one with the lowest chi2
    min_chi2 = jnp.inf
    for i in range(iter_init):
        log.info(f"Hessian fit initialization iteration {i+1}")
        # Generate random initial parameters
        initial_parameters = pdf_initial_parameters(
            pdf_model, param_initialiser_settings, i
        )

        # Delegate to generic gradient descent
        gd_result_iter = run_gradient_descent(
            initial_parameters=initial_parameters,
            training_loss_fn=train_chi2,
            validation_loss_fn=valid_chi2,
            optimizer=optimizer_provider,
            early_stopper=early_stopper,
            max_epochs=max_epochs,
            data_batch=None,
            record_every=50,
        )
        parameters_min_iter = gd_result_iter.optimized_parameters
        min_chi2_iter = train_chi2(parameters_min_iter, 0)

        if min_chi2_iter < min_chi2:
            min_chi2 = min_chi2_iter
            gd_result = gd_result_iter

    log.info(f"Minimum chi2 found: {min_chi2}")
    # Extract results from the best fit
    parameters_min = gd_result.optimized_parameters
    training_loss = gd_result.training_loss

    # Compute gradient at the candidate minimum to check stationarity
    grad_at_min = jax.grad(lambda p: train_chi2(p, 0))(parameters_min)
    grad_norm = jnp.linalg.norm(grad_at_min)
    if jnp.isnan(grad_norm) | jnp.isinf(grad_norm):
        log.critical(
            "Gradient norm at the found minimum is NaN/Inf; local minimum check cannot be trusted."
        )
    # Compute the Hessian matrix at the minimum, using the train_chi2 function
    # giving the optimized parameters and idx=0 (not used)
    # Note the 0.5 factor because we absorb the 1/2 of the taylor expansion
    # in the definition of the Hessian
    hessian = 0.5 * jax.hessian(lambda p: train_chi2(p, 0))(parameters_min)

    # Verify hessian is positive definite
    eigvals = jnp.linalg.eigvalsh(hessian)
    min_eigval = jnp.min(eigvals)
    if min_eigval <= 0:
        log.critical(
            f"WARNING: The Hessian matrix is not positive definite. "
            f"Minimum eigenvalue is {min_eigval:.2e}."
        )

    # Local minimum check: small gradient and positive-definite Hessian
    is_grad_small = grad_norm <= grad_tol
    is_pd = min_eigval > eig_eps
    if is_grad_small and is_pd:
        log.info(
            f"Local minimum verified: ||grad||={float(grad_norm):.3e}<= {grad_tol:.1e}, "
            f"min_eig={float(min_eigval):.3e} > {eig_eps:.1e}."
        )
    else:
        log.critical(
            "Local minimum check failed: "
            f"||grad||={float(grad_norm):.3e} (tol {grad_tol:.1e}), "
            f"min_eig={float(min_eigval):.3e} (must be > {eig_eps:.1e})."
        )
        if require_local_min:
            raise ValueError(
                "Hessian fit did not converge to a verified local minimum (stationary point with PD Hessian)."
            )

    # covariance matrix is inflated by tolerance^2
    cov_params = tolerance**2 * jnp.linalg.inv(hessian)

    t1 = time.time()
    log.info("HESSIAN RUNNING TIME: %f" % (t1 - t0))

    if hessian_settings["ErrorType"] == "replicas":
        log.info("Using gaussian replicas for error propagation.")
        n_samples = hessian_settings["n_samples"]

        # Generate samples from a multivariate normal distribution
        # with mean parameters_min and covariance cov_params
        hessian_param_set = jax.random.multivariate_normal(
            key=jax.random.PRNGKey(0),
            mean=parameters_min,
            cov=cov_params,
            shape=(n_samples,),
        )
    elif hessian_settings["ErrorType"] == "hessian":
        log.info("Using hessian method for error propagation.")
        # Find eingenvectors and eigenvalues of the covariance matrix
        eigvals, eigvecs = jnp.linalg.eigh(cov_params)
        # Generate symmetric parameter variations along each eigenvector
        # scaled by sqrt of the corresponding eigenvalue
        param_shift = (eigvecs * jnp.sqrt(eigvals)).T  # (n_eig, n_par)
        # Shift the optimized parameters along the eigenvectors
        # to generate the error sets
        param_plus = parameters_min + param_shift
        param_minus = parameters_min - param_shift
        hessian_param_set = jnp.empty(
            (2 * param_plus.shape[0], param_plus.shape[1]), dtype=param_plus.dtype
        )
        hessian_param_set = hessian_param_set.at[0::2].set(param_plus)  # even rows: +
        hessian_param_set = hessian_param_set.at[1::2].set(param_minus)  # odd  rows: −
    else:
        raise ValueError(
            f"Unknown ErrorType {hessian_settings['ErrorType']}. Choose 'replicas' or 'hessian'."
        )

    return HessianFit(
        hessian_specs={
            "max_epochs": max_epochs,
            "tolerance": tolerance,
            "iter_init": iter_init,
            "grad_tol": grad_tol,
            "min_hessian_eigval": eig_eps,
            "require_local_min": require_local_min,
            "ErrorType": hessian_settings["ErrorType"],
        },
        min_chi2=min_chi2,
        training_loss=training_loss,
        optimized_parameters=parameters_min,
        hessian=hessian,
        cov_params=cov_params,
        resampled_posterior=hessian_param_set,
    )


def run_hessian_fit(hessian_fit, output_path, pdf_model):
    """
    Export the results of a Hessian fit.

    Parameters
    ----------
    hessian_fit: HessianFit
        The results of the Hessian fit.
    output_path: pathlib.PosixPath
        Path to the output folder.
    pdf_model: pdf_model.PDFModel
        The PDF model used in the fit.
    """

    # export_hessian_results(hessian_fit, output_path, "hessian_result")

    write_replicas(hessian_fit, output_path, pdf_model)
