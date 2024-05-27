"""
wmin.utils is a module that contains several utils for PDF fits in
the wmin parameterization.
"""

import jax
import time
import pandas as pd
from reportengine.table import table



@table
def likelihood_time(
    _chi2_with_positivity,
    _pred_data,
    FIT_XGRID,
    pdf_model,
    bayesian_prior,
    data,
    n_prior_samples=1000,
):
    """
    TODO
    """
    ndata = sum([ds.load_commondata().ndata for ds in data.datasets])

    pred_and_pdf = pdf_model.pred_and_pdf_func(FIT_XGRID, forward_map=_pred_data)

    @jax.jit
    def log_likelihood(params):
        predictions, pdf = pred_and_pdf(params)
        return -0.5 * _chi2_with_positivity(predictions, pdf)

    # sample from prior
    rng = jax.random.PRNGKey(0)
    prior_samples = []
    for i in range(n_prior_samples):
        prior_samples.append(
            bayesian_prior(jax.random.uniform(rng, shape=(pdf_model.n_basis,)))
        )

    # compile likelihood
    log_likelihood(prior_samples[0])

    # evaluate likelihood time
    start_time = time.perf_counter()
    for i in range(n_prior_samples):
        log_likelihood(prior_samples[i])
    end_time = time.perf_counter()

    time_per_eval = (end_time - start_time) / n_prior_samples

    df = pd.DataFrame({"Ndata": [ndata], "Likelihood eval time (s)": [time_per_eval]}, index=["wmin"])
    return df
