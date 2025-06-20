"""
colibri.ultranest_fit.py

This module contains the main Bayesian fitting routine of colibri.

"""

import logging
import sys
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import blackjax
from blackjax.ns.utils import finalise, sample, log_weights, ess
from jax.scipy.special import logsumexp
import tqdm
from colibri.export_results import BayesianFit, export_bayes_results, write_replicas
from colibri.utils import resample_from_ns_posterior
import anesthetic


log = logging.getLogger(__name__)

# Check if --debug flag is present
debug_flag = "--debug" in sys.argv

# Set the Ultranest logging level based on the presence of --debug flag
ultranest_logger = logging.getLogger("blackjax")
ultranest_logger.setLevel(logging.DEBUG if debug_flag else logging.INFO)

# Configure the handler and formatter
handler = logging.StreamHandler(sys.stdout)
ultranest_logger.addHandler(handler)


@dataclass(frozen=True)
class NSSFit(BayesianFit):
    """
    Dataclass containing the results and specs of an Ultranest fit.

    Attributes
    ----------
    ultranest_specs: dict
        Dictionary containing the settings of the Ultranest fit.
    ultranest_result: dict
        result from ultranest, can be used eg for corner plots
    """

    nss_specs: dict
    nss_result: dict


def blackjax_fit(
    pdf_model,
    bayesian_prior,
    ns_settings,
    log_likelihood,
):
    """
    The complete Nested Sampling fitting routine, for any PDF model.

    Parameters
    ----------
    pdf_model: pdf_model.PDFModel
        The PDF model to fit.

    bayesian_prior: @jax.jit CompiledFunction
        The prior function for the model.

    ns_settings: dict
        Settings for the Nested Sampling fit.

    log_likelihood: Callable
        The log likelihood function for the model.

    Returns
    -------
    UltranestFit
        Dataclass containing the results and specs of an Ultranest fit.
    """

    log.info(f"Running fit with backend: {jax.lib.xla_bridge.get_backend().platform}")

    # set the ultranest seed
    rng_key = jax.random.PRNGKey(ns_settings["blackjax_settings"]["seed"])
    n_dims = pdf_model.n_parameters
    n_live = ns_settings["blackjax_settings"]["n_live"]
    n_delete = int(ns_settings["blackjax_settings"]["delete_fraction"] * n_live )

    inital_particles = bayesian_prior["sample"](
        rng_key, n_live
    )

    algo = blackjax.nss(
        logprior_fn=bayesian_prior["log_prob"],
        loglikelihood_fn=log_likelihood,
        num_delete=n_delete,
        num_inner_steps=int(ns_settings["blackjax_settings"]["repeats"] * n_dims),
    )

    @jax.jit
    def one_step(carry, xs):
        state, k = carry
        k, subk = jax.random.split(k, 2)
        state, dead_point = algo.step(subk, state)
        return (state, k), dead_point

    state = algo.init(inital_particles)

    dead = []


    t0 = time.time()
    with tqdm.tqdm(desc="Dead points", unit=" dead points") as pbar:
        while not state.logZ_live - state.logZ < ns_settings["blackjax_settings"]["log_precision"]:
            (state, rng_key), dead_info = one_step((state, rng_key), None)
            dead.append(dead_info)
            pbar.update(n_delete)
    t1 = time.time()

    log.info("BLACKJAX NSS RUNNING TIME: %f" % (t1 - t0))

    final_states = finalise(
        state,
        dead
    )
    rng_key, ess_key, weights_key, sample_key = jax.random.split(rng_key, 4)

    # Initialize fit_result to avoid UnboundLocalError
    fit_result = None

    ess_value = int(ess(ess_key,final_states))
    logw = log_weights(rng_key,final_states)
    logzs = logsumexp(logw, axis =0)
    samples = sample(sample_key, final_states, ess_value)
    

    #write out an anesthetic dataframe
    nested_samples = anesthetic.NestedSamples(
        data = final_states.particles,
        logL = final_states.loglikelihood,
        logL_birth = final_states.loglikelihood_birth,
        columns = pdf_model.param_names,
    )
    # piggy back on ultranest log dir
    nested_samples.to_csv(ns_settings["ReactiveNS_settings"]["log_dir"] + "/nested_samples.csv")

    #todo: interface properly to expected output
    fit_result = NSSFit(
        param_names=pdf_model.param_names,
        resampled_posterior=samples,
        full_posterior_samples=samples,
        nss_specs=ns_settings,
        bayesian_metrics={},
        nss_result={
            "logZ": logzs.mean(),
            "logZ_err": logzs.std(),
            "ess": ess_value,
        }
    )

    return fit_result


def run_blackjax_fit(blackjax_fit, output_path, pdf_model):
    """
    Export the results of an Ultranest fit.

    Parameters
    ----------
    ultranest_fit: UltranestFit
        The results of the Ultranest fit.
    output_path: pathlib.PosixPath
        Path to the output folder.
    pdf_model: pdf_model.PDFModel
        The PDF model used in the fit.
    """

    export_bayes_results(blackjax_fit, output_path, "ns_result")

    write_replicas(blackjax_fit, output_path, pdf_model)
