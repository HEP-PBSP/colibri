from colibri.export_results import write_replicas

# from colibri.export_results import export_hessian_results
import jax.numpy as jnp
from dataclasses import dataclass
import logging
import time
import jax
from colibri.gradient_descent import run_gradient_descent
from colibri.mc_initialisation import mc_initial_parameters

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
    mc_initialiser_settings,
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

    t0 = time.time()
    # Generate iter_init random initial parameters
    # run the fit and pick the one with the lowest chi2
    min_chi2 = jnp.inf
    for i in range(iter_init):
        log.info(f"Hessian fit initialization iteration {i+1}")
        # Generate random initial parameters
        initial_parameters = mc_initial_parameters(
            pdf_model, mc_initialiser_settings, i
        )

        # Delegate to generic gradient descent
        gd_result = run_gradient_descent(
            initial_parameters=initial_parameters,
            training_loss_fn=train_chi2,
            validation_loss_fn=valid_chi2,
            optimizer=optimizer_provider,
            early_stopper=early_stopper,
            max_epochs=max_epochs,
            data_batch=None,
            record_every=50,
        )
        parameters_min_iter = gd_result.optimized_parameters
        min_chi2_iter = train_chi2(parameters_min_iter, 0)

        if min_chi2_iter < min_chi2:
            min_chi2 = min_chi2_iter
            parameters_min = parameters_min_iter
            training_loss = gd_result.training_loss

    log.info(f"Minimum chi2 found: {min_chi2}")
    # Compute the Hessian matrix at the minimum, using the train_chi2 function
    # giving the optimized parameters and idx=0 (not used)
    # Note the 0.5 factor because we absorb the 1/2 of the taylor expansion
    # in the definition of the Hessian
    hessian = 0.5 * jax.hessian(train_chi2)(parameters_min, 0)

    # Verify hessian is positive definite
    eigvals = jnp.linalg.eigvalsh(hessian)
    min_eigval = jnp.min(eigvals)
    if min_eigval <= 0:
        log.critical(
            f"WARNING: The Hessian matrix is not positive definite. "
            f"Minimum eigenvalue is {min_eigval:.2e}."
        )

    # covariance matrix is inflated by tolerance^2
    cov_params = tolerance**2 * jnp.linalg.inv(hessian)

    n_samples = hessian_settings["n_samples"]

    # Generate samples from a multivariate normal distribution
    # with mean parameters_min and covariance cov_params
    resampled_posterior = jax.random.multivariate_normal(
        key=jax.random.PRNGKey(0),
        mean=parameters_min,
        cov=cov_params,
        shape=(n_samples,),
    )

    t1 = time.time()
    log.info("HESSIAN RUNNING TIME: %f" % (t1 - t0))

    return HessianFit(
        hessian_specs={
            "max_epochs": max_epochs,
            "tolerance": tolerance,
            "iter_init": iter_init,
        },
        min_chi2=min_chi2,
        training_loss=training_loss,
        optimized_parameters=parameters_min,
        hessian=hessian,
        cov_params=cov_params,
        resampled_posterior=resampled_posterior,
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
