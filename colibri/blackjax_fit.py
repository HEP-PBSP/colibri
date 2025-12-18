"""
colibri.blackjax_fit.py

This module contains the BlackJAX Bayesian fitting routine of Colibri.

"""

import logging
import sys
import time
import os

import jax
import jax.numpy as jnp
import blackjax
from blackjax.ns.utils import finalise, sample, log_weights, ess
from jax.scipy.special import logsumexp
import tqdm
import anesthetic

from colibri.core import BlackJAXFit
from colibri.export_results import export_bayes_results, write_replicas
from colibri.utils import resample_from_ns_posterior


log = logging.getLogger(__name__)

# Check if --debug flag is present
debug_flag = "--debug" in sys.argv

# Set the BlackJAX logging level based on the presence of --debug flag
blackjax_logger = logging.getLogger("blackjax")
blackjax_logger.setLevel(logging.DEBUG if debug_flag else logging.INFO)

# Configure the handler and formatter
handler = logging.StreamHandler(sys.stdout)
blackjax_logger.addHandler(handler)


def blackjax_fit(
    pdf_model,
    bayesian_prior,
    blackjax_settings,
    log_likelihood,
):
    """
    The complete Nested Sampling fitting routine using BlackJAX, for any PDF model.

    Parameters
    ----------
    pdf_model: pdf_model.PDFModel
        The PDF model to fit.

    bayesian_prior: @jax.jit CompiledFunction
        The prior function for the model.

    blackjax_settings: dict
        Settings for the BlackJAX Nested Sampling fit.

    log_likelihood: Callable
        The log likelihood function for the model.

    Returns
    -------
    BlackJAXFit
        Dataclass containing the results and specs of a BlackJAX fit.
    """

    log.info(f"Running fit with backend: {jax.lib.xla_bridge.get_backend().platform}")

    # set the BlackJAX seed
    rng_key = jax.random.PRNGKey(blackjax_settings["seed"])
    log.info(f"BlackJAX initialisation seed: {rng_key}")
    n_dims = pdf_model.n_parameters
    n_live = blackjax_settings["n_live"]
    n_delete = int(blackjax_settings["delete_fraction"] * n_live)

    inital_particles = bayesian_prior["sample"](rng_key, n_live)

    algo = blackjax.nss(
        logprior_fn=bayesian_prior["log_prob"],
        loglikelihood_fn=log_likelihood,
        num_delete=n_delete,
        num_inner_steps=int(blackjax_settings["repeats"] * n_dims),
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
        while not state.logZ_live - state.logZ < blackjax_settings["log_precision"]:
            (state, rng_key), dead_info = one_step((state, rng_key), None)
            dead.append(dead_info)
            pbar.update(n_delete)
    t1 = time.time()

    log.info("BLACKJAX RUNNING TIME: %f" % (t1 - t0))

    final_states = finalise(state, dead)
    rng_key, ess_key, weights_key, sample_key = jax.random.split(rng_key, 4)

    # Initialize fit_result to avoid UnboundLocalError
    fit_result = None

    ess_value = int(ess(ess_key, final_states))
    logw = log_weights(rng_key, final_states)
    logzs = logsumexp(logw, axis=0)
    full_samples = sample(sample_key, final_states, ess_value)

    # Get number of posterior samples to resample
    n_posterior_samples = blackjax_settings["n_posterior_samples"]

    # Check if we have enough samples
    if n_posterior_samples > full_samples.shape[0]:
        n_posterior_samples = full_samples.shape[0]
        log.warning(
            f"The chosen number of posterior samples exceeds the number of posterior "
            f"samples computed by BlackJAX. Setting the number of resampled posterior "
            f"samples to {n_posterior_samples}"
        )

    # Resample the posterior
    posterior_resampling_seed = blackjax_settings["posterior_resampling_seed"]
    log.info(f"Resampling posterior with seed: {posterior_resampling_seed}")
    resampled_posterior = resample_from_ns_posterior(
        full_samples,
        n_posterior_samples,
        posterior_resampling_seed,
    )

    # write out an anesthetic dataframe
    nested_samples = anesthetic.NestedSamples(
        data=final_states.particles,
        logL=final_states.loglikelihood,
        logL_birth=final_states.loglikelihood_birth,
        columns=pdf_model.param_names,
    )
    # write nested_samples.csv to blackjax_logs
    log_dir = blackjax_settings["log_dir"]
    os.makedirs(log_dir, exist_ok=True)  # Create directory if it doesn't exist
    nested_samples.to_csv(log_dir + "/nested_samples.csv")

    # Compute bayesian metrics (similar to UltraNest)
    # Find maximum likelihood point
    max_ll_idx = jnp.argmax(final_states.loglikelihood)
    min_chi2 = -2 * final_states.loglikelihood[max_ll_idx]

    # Compute average chi2 over full samples
    avg_chi2 = jnp.array(
        [-2 * log_likelihood(jnp.array(sample)).item() for sample in full_samples]
    ).mean()

    Cb = avg_chi2 - min_chi2

    # todo: interface properly to expected output
    fit_result = BlackJAXFit(
        blackjax_specs=blackjax_settings,
        blackjax_result={
            "logZ": logzs.mean(),
            "logZ_err": logzs.std(),
            "ess": ess_value,
        },
        param_names=pdf_model.param_names,
        resampled_posterior=resampled_posterior,
        full_posterior_samples=full_samples,
        bayesian_metrics={
            "bayes_complexity": Cb,
            "avg_chi2": avg_chi2,
            "min_chi2": min_chi2,
            "logz": logzs.mean(),
        },
    )

    return fit_result


def run_blackjax_fit(blackjax_fit, output_path, pdf_model):
    """
    Export the results of a BlackJAX fit.

    Parameters
    ----------
    blackjax_fit: BlackJAXFit
        The results of the BlackJAX fit.
    output_path: pathlib.PosixPath
        Path to the output folder.
    pdf_model: pdf_model.PDFModel
        The PDF model used in the fit.
    """

    export_bayes_results(blackjax_fit, output_path, "ns_result")

    write_replicas(blackjax_fit, output_path, pdf_model)
