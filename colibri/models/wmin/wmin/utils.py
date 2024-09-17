"""
wmin.utils is a module that contains several utils for PDF fits in
the wmin parameterization.
"""

import time
import resource
import logging

import jax
import pandas as pd
from colibri.loss_functions import chi2
from colibri.ultranest_fit import UltraNestLogLikelihood
from reportengine.table import table


log = logging.getLogger(__name__)


"""
Memory usage before loading of resources
"""
RSS_MB = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
init = RSS_MB()


@table
def likelihood_time(
    _penalty_posdata,
    central_inv_covmat_index,
    fast_kernel_arrays,
    positivity_fast_kernel_arrays,
    _pred_data,
    FIT_XGRID,
    pdf_model,
    bayesian_prior,
    theoryid,
    ns_settings,
    n_prior_samples=1000,
    positivity_penalty_settings={},
):
    """
    This function calculates the time it takes to evaluate the likelihood
    function.

    Parameters
    ----------
    _penalty_posdata: function
        The positivity penalty function to use.

    central_inv_covmat_index: colibri.commondata_utils.CentralInvCovmatIndex

    fast_kernel_arrays: tuple
        The FK tables to use.

    positivity_fast_kernel_arrays: tuple
        The POS FK tables to use.

    _pred_data: function
        The prediction function to use.

    FIT_XGRID: array
        The xgrid to use.

    pdf_model: PDFModel

    bayesian_prior: function
        The prior function to use.

    theoryid: str

    ns_settings: dict

    n_prior_samples: int, 1000

    alpha: float, 1e-7

    lambda_positivity: int, 1000

    Returns
    -------
    df: pd.DataFrame
        The DataFrame containing the results.
    """
    log.info(f"Memory usage after loading of resources")
    res = RSS_MB()
    log.info(f"RSS: {res - init:.2f}MB")

    central_values = central_inv_covmat_index.central_values
    ndata = len(central_values)

    log_likelihood = UltraNestLogLikelihood(
        central_inv_covmat_index,
        pdf_model,
        FIT_XGRID,
        _pred_data,
        fast_kernel_arrays,
        positivity_fast_kernel_arrays,
        ns_settings,
        chi2,
        _penalty_posdata,
        positivity_penalty_settings,
    )

    # sample from prior
    rng = jax.random.PRNGKey(0)
    prior_samples = []
    for i in range(n_prior_samples):
        prior_samples.append(
            bayesian_prior(jax.random.uniform(rng, shape=(pdf_model.n_basis,)))
        )

    # compile likelihood
    log_likelihood(prior_samples[0])

    log.info(f"Memory usage after initialization of log_likelihood")
    log.info(f"RSS: {RSS_MB() - res:.2f}MB")
    # evaluate likelihood time
    start_time = time.perf_counter()
    for i in range(n_prior_samples):
        log_likelihood(prior_samples[i])
    end_time = time.perf_counter()

    time_per_eval = (end_time - start_time) / n_prior_samples

    df = pd.DataFrame(
        {
            "Ndata": [ndata],
            "Theory": [theoryid],
            "Likelihood eval time (s)": [time_per_eval],
        },
        index=["wmin"],
    )
    return df
