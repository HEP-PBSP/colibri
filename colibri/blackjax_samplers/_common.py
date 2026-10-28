"""Shared settings and output contract for BlackJAX samplers."""

import os
from dataclasses import dataclass, field
from numbers import Integral
from typing import Any, Optional

COMMON_SETTINGS = {
    "blackjax_seed": 0,
    "n_posterior_samples": 1000,
    "posterior_resampling_seed": 123456,
    "sampler_plot": True,
}


@dataclass(frozen=True)
class SamplerOutput:
    """Results returned by a sampler before common fit processing.

    samples:
        Equal-weight physical parameter draws, with shape
        (n_draws, n_parameters).

    log_likelihoods:
        Physical log-likelihood values corresponding to samples,
        with shape (n_draws,).

    max_loglikelihood:
        Best likelihood encountered by the sampler. This may come
        from a point outside the returned posterior draws.

    logz, logz_err:
        Evidence estimate and uncertainty, where available.

    resampled_posterior:
        Optional sampler-selected replicas. If omitted, the shared provider
        randomly selects replicas from samples. NUTS supplies thinned draws.

    result:
        Algorithm-specific state, samples, and diagnostics.
    """

    samples: Any
    log_likelihoods: Any
    max_loglikelihood: Any
    logz: Optional[Any] = None
    logz_err: Optional[Any] = None
    result: dict = field(default_factory=dict)
    resampled_posterior: Optional[Any] = None


def validate_common_settings(settings):
    """Validate settings shared by all BlackJAX samplers."""
    for name in ("blackjax_seed", "posterior_resampling_seed"):
        value = settings[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, Integral)
            or not 0 <= value < 2**32
        ):
            raise ValueError(f"{name} must be an integer in [0, 2**32).")

    count = settings["n_posterior_samples"]
    if isinstance(count, bool) or not isinstance(count, Integral) or count <= 0:
        raise ValueError("n_posterior_samples must be a positive integer.")

    if not isinstance(settings["sampler_plot"], bool):
        raise ValueError("sampler_plot must be a boolean.")

    if "log_dir" in settings:
        path = settings["log_dir"]
        if not isinstance(path, (str, os.PathLike)):
            raise ValueError("log_dir must be a filesystem path.")

        if not os.fspath(path):
            raise ValueError("log_dir cannot be empty.")
