"""
colibri.config_utils.py

Modules contains utility functions for config.py module.
"""

import logging
from reportengine.configparser import ConfigError
import os

from colibri.core import ColibriNestedSamplingSpecs


log = logging.getLogger(__name__)


def ns_settings_parser(
    settings,
    output_path,
):
    """
    For a Nested Sampling fit, parses the ns_settings namespace from the runcard,
    and ensures the choice of settings is valid.
    """

    # Begin by checking that the user-supplied keys are known; warn the user otherwise.
    known_keys = {
        "n_posterior_samples",
        "posterior_resampling_seed",
        "ReactiveNS_settings",
        "Run_settings",
        "SliceSampler_settings",
        "ultranest_seed",
        "sampler_plot",
        "popstepsampler",
    }

    kdiff = settings.keys() - known_keys
    for k in kdiff:
        log.warning(ConfigError(f"Key '{k}' in ns_settings not known.", k, known_keys))

    # Now construct the ns_settings dictionary, checking the parameter combinations are
    # valid
    ns_settings = {}

    # Set the ultranest seed
    ns_settings["ultranest_seed"] = settings.get("ultranest_seed", 123456)

    # Set the posterior resampling parameters
    ns_settings["n_posterior_samples"] = settings.get("n_posterior_samples", 1000)
    ns_settings["posterior_resampling_seed"] = settings.get(
        "posterior_resampling_seed", 123456
    )

    # Parse internal settings, if they are not mentioned, set to empty dict
    ns_settings["ReactiveNS_settings"] = settings.get("ReactiveNS_settings", {})
    ns_settings["Run_settings"] = settings.get("Run_settings", {})
    ns_settings["SliceSampler_settings"] = settings.get("SliceSampler_settings", {})

    # set sampler plot to True by default
    ns_settings["sampler_plot"] = settings.get("sampler_plot", True)

    # set popstepsampler to False by default
    ns_settings["popstepsampler"] = settings.get("popstepsampler", False)

    # Check that the ReactiveNS_settings key was provided, if not set to default
    if ns_settings["ReactiveNS_settings"]:
        # Set the directory where the ultranest logs will be stored; by default
        # they are stored in output_path/ultranest_logs
        ns_settings["ReactiveNS_settings"]["log_dir"] = settings[
            "ReactiveNS_settings"
        ].get("log_dir", str(output_path / "ultranest_logs"))

        ns_settings["ReactiveNS_settings"]["resume"] = settings[
            "ReactiveNS_settings"
        ].get("resume", False)

        ns_settings["ReactiveNS_settings"]["vectorized"] = settings[
            "ReactiveNS_settings"
        ].get("vectorized", False)
    else:
        ns_settings["ReactiveNS_settings"]["log_dir"] = str(
            output_path / "ultranest_logs"
        )
        ns_settings["ReactiveNS_settings"]["resume"] = False
        ns_settings["ReactiveNS_settings"]["vectorized"] = False

    # In the case that the fit is resuming from a previous ultranest fit, the logs
    # directory must exist
    if ns_settings["ReactiveNS_settings"]["resume"]:
        if not os.path.exists(ns_settings["ReactiveNS_settings"]["log_dir"]):
            raise FileNotFoundError(
                "Could not find previous ultranest fit at "
                + str(ns_settings["ReactiveNS_settings"]["log_dir"])
                + "."
            )

        log.info(
            "Resuming ultranest fit from "
            + str(ns_settings["ReactiveNS_settings"]["log_dir"])
            + "."
        )

    # If the resume option is false, ultranest expects "overwrite" instead
    if not ns_settings["ReactiveNS_settings"]["resume"]:
        ns_settings["ReactiveNS_settings"]["resume"] = "overwrite"

    return ColibriNestedSamplingSpecs(ns_settings=ns_settings)


def analytic_settings_parser(
    settings,
):
    """
    For an analytic fit, parses the analytic_settings namespace from the runcard,
    and ensures the choice of settings is valid.
    """

    # Begin by checking that the user-supplied keys are known; warn the user otherwise.
    known_keys = {
        "n_posterior_samples",
        "sampling_seed",
        "full_sample_size",
        "optimal_prior",
    }

    kdiff = settings.keys() - known_keys
    for k in kdiff:
        log.warning(
            ConfigError(f"Key '{k}' in analytic_settings not known.", k, known_keys)
        )

    # Now construct the analytic_settings dictionary, checking the parameter combinations are
    # valid
    analytic_settings = {}

    # Set the sampling seed
    analytic_settings["sampling_seed"] = settings.get("sampling_seed", 123456)

    # Set the posterior resampling parameters
    analytic_settings["n_posterior_samples"] = settings.get("n_posterior_samples", 100)

    # Set the full sample size
    analytic_settings["full_sample_size"] = settings.get("full_sample_size", 1000)

    # Set the optimal prior flag
    analytic_settings["optimal_prior"] = settings.get("optimal_prior", False)

    return analytic_settings
