"""BlackJAX nested sampling and its diagnostic log export."""

import logging
import time
from numbers import Integral, Real
from pathlib import Path

import blackjax
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from blackjax.ns.utils import ess, finalise, log_weights, sample
from jax.scipy.special import logsumexp

from ._common import SamplerOutput

log = logging.getLogger(__name__)

SETTINGS = {
    "n_live": 500,
    "repeats": 3,
    "delete_fraction": 0.5,
    "log_precision": -3,
}

RESULT_FILENAME = "ns_result"


def validate(settings):
    """Validate nested-sampling settings."""
    n_live = settings["n_live"]
    if isinstance(n_live, bool) or not isinstance(n_live, Integral) or n_live < 2:
        raise ValueError("n_live must be an integer of at least two.")

    for name in ("repeats", "delete_fraction", "log_precision"):
        value = settings[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not np.isfinite(value)
        ):
            raise ValueError(f"{name} must be a finite number.")

    if settings["repeats"] < 1:
        raise ValueError("repeats must be at least one.")

    fraction = settings["delete_fraction"]
    if not 0 < fraction < 1:
        raise ValueError("delete_fraction must be strictly between zero and one.")

    if int(fraction * n_live) < 1:
        raise ValueError("delete_fraction must delete at least one live point.")


def run(
    rng_key,
    prior,
    log_likelihood,
    settings,
    *,
    param_names,
):
    """Return equal-weight posterior draws and nested-sampling state."""
    n_dims = len(param_names)
    n_live = settings["n_live"]
    n_delete = int(settings["delete_fraction"] * n_live)

    initial_particles = prior.sample(rng_key, n_live)

    algorithm = blackjax.nss(
        logprior_fn=prior.log_prob,
        loglikelihood_fn=log_likelihood,
        num_delete=n_delete,
        num_inner_steps=int(settings["repeats"] * n_dims),
    )

    @jax.jit
    def one_step(carry):
        state, key = carry
        key, step_key = jax.random.split(key)

        state, dead_points = algorithm.step(
            step_key,
            state,
        )
        return (state, key), dead_points

    state = algorithm.init(initial_particles)
    dead = []

    started = time.perf_counter()

    with tqdm.tqdm(
        desc="Dead points",
        unit=" dead points",
    ) as progress:
        while not (state.logZ_live - state.logZ < settings["log_precision"]):
            (state, rng_key), dead_points = one_step((state, rng_key))
            dead.append(dead_points)
            progress.update(n_delete)

    log.info(
        "BlackJAX nested sampling finished in %.2f seconds.",
        time.perf_counter() - started,
    )

    final_states = finalise(state, dead)

    # Preserve the random-key schedule of the existing implementation.
    rng_key, ess_key, _weights_key, sample_key = jax.random.split(
        rng_key,
        4,
    )

    ess_value = int(ess(ess_key, final_states))
    if ess_value <= 0:
        raise RuntimeError("Nested sampling produced no effective posterior draws.")

    logw = log_weights(rng_key, final_states)
    logzs = logsumexp(logw, axis=0)

    full_samples = sample(
        sample_key,
        final_states,
        ess_value,
    )

    # Preserve sequential likelihood evaluation from the existing fit.
    log_likelihoods = jnp.asarray([log_likelihood(point) for point in full_samples])

    best_index = jnp.argmax(final_states.loglikelihood)

    return SamplerOutput(
        samples=full_samples,
        log_likelihoods=log_likelihoods,
        max_loglikelihood=final_states.loglikelihood[best_index],
        logz=logzs.mean(),
        logz_err=logzs.std(),
        result={
            "ess": ess_value,
            "best_point": final_states.particles[best_index],
            "nested_particles": final_states.particles,
            "nested_loglikelihood": final_states.loglikelihood,
            "nested_loglikelihood_birth": (final_states.loglikelihood_birth),
        },
    )


def export_logs(fit, log_dir):
    """Write nested particles in Anesthetic format."""
    import anesthetic

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    result = fit.blackjax_result

    nested_samples = anesthetic.NestedSamples(
        data=result["nested_particles"],
        logL=result["nested_loglikelihood"],
        logL_birth=result["nested_loglikelihood_birth"],
        columns=fit.param_names,
    )

    nested_samples.to_csv(log_dir / "nested_samples.csv")
