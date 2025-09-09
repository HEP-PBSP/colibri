from colibri.export_results import write_replicas

# from colibri.export_results import export_hessian_results
import jax.numpy as jnp
from dataclasses import dataclass
import logging
import time
import jax
from colibri.gradient_descent import run_gradient_descent

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
    optimized_parameters: jnp.ndarray
        Array containing the optimized parameters in the minimum of the chi2.
    hessian: jnp.ndarray
        Array containing the Hessian matrix at the minimum of the chi2.
    """

    hessian_specs: dict
    min_chi2: jnp.ndarray
    optimized_parameters: jnp.ndarray
    hessian: jnp.ndarray


def hessian_fit(
    pdf_model,
    log_likelihood,
    optimizer_provider,
    early_stopper,
    max_epochs,
    hessian_settings,
):

    log.info(f"Running fit with backend: {jax.lib.xla_bridge.get_backend().platform}")
    log.info("Starting Hessian fit...")

    # run_gradient_descent expects a data batch object, but we don't use it here
    def train_chi2(params, idx):
        return -2 * log_likelihood(params)

    def valid_chi2(params):
        return jnp.nan

    len_params = len(pdf_model.param_names)
    # Generate random initial parameters
    initial_parameters = jax.random.uniform(jax.random.PRNGKey(0), (len_params,))
    tolerance = hessian_settings["tolerance"]
    t0 = time.time()

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

    t1 = time.time()
    log.info("HESSIAN RUNNING TIME: %f" % (t1 - t0))

    return HessianFit(
        hessian_specs={
            "max_epochs": max_epochs,
            "tolerance": tolerance,
        },
        min_chi2=gd_result.training_loss,
        optimized_parameters=gd_result.optimized_parameters,
        hessian=None,
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

    # write_replicas(hessian_fit, output_path, pdf_model)
    pass
