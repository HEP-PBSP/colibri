"""
colibri.config.py

Config module of colibri

Author: Mark N. Costantini
Note: several functions are taken from validphys.config
Date: 11.11.2023
"""

from validphys.config import Config, Environment
from validphys import covmats

from colibri import covmats as colibri_covmats

from reportengine.configparser import explicit_node, ConfigError

from colibri import commondata_utils

from colibri.constants import FLAVOUR_TO_ID_MAPPING

import logging
import os

log = logging.getLogger(__name__)


class Environment(Environment):
    def __init__(self, replica_index=None, trval_index=0, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.replica_index = replica_index
        self.trval_index = trval_index

    @classmethod
    def ns_dump_description(cls):
        return {
            "replica_index": "The MC replica index",
            "trval_index": "The Training/Validation split index",
            **super().ns_dump_description(),
        }


class colibriConfig(Config):
    """
    Config class inherits from validphys
    Config class
    """

    def parse_ns_settings(
        self,
        settings,
        output_path,
    ):
        """For a Nested Sampling fit, parses the ns_settings namespace from the runcard,
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
        }

        kdiff = settings.keys() - known_keys
        for k in kdiff:
            log.warning(
                ConfigError(f"Key '{k}' in ns_settings not known.", k, known_keys)
            )

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
        else:
            ns_settings["ReactiveNS_settings"]["log_dir"] = str(
                output_path / "ultranest_logs"
            )
            ns_settings["ReactiveNS_settings"]["resume"] = False

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

        return ns_settings

    @explicit_node
    def produce_commondata_tuple(self, closure_test_level=False):
        """
        Produces a commondata tuple node in the reportengine dag
        according to some options
        """
        if closure_test_level is False:
            return commondata_utils.experimental_commondata_tuple
        elif closure_test_level == 0:
            return commondata_utils.level_0_commondata_tuple
        elif closure_test_level == 1:
            return commondata_utils.level_1_commondata_tuple
        else:
            raise ValueError(
                "closure_test_level must be either False, 0 or 1, if not specified in the runcard then Experimental data is used."
            )

    @explicit_node
    def produce_fit_covariance_matrix(self, use_fit_t0: bool = True):
        """
        Produces the covariance matrix used in the fit.
        This covariance matrix is used in:
            - commondata_utils.central_covmat_index
            - loss functions in mc_loss_functions.py
        """
        if use_fit_t0:
            return colibri_covmats.dataset_inputs_t0_covmat_from_systematics
        else:
            return colibri_covmats.dataset_inputs_covmat_from_systematics

    @explicit_node
    def produce_data_generation_covariance_matrix(self, use_gen_t0: bool = False):
        """Produces the covariance matrix used in:
        - level 1 closure test data construction (fluctuating around the level
        0 data)
        - Monte Carlo pseudodata (fluctuating either around the level 0 data or
        level 1 data)
        """
        if use_gen_t0:
            return colibri_covmats.dataset_inputs_t0_covmat_from_systematics
        else:
            return colibri_covmats.dataset_inputs_covmat_from_systematics

    def produce_replica_indices(self, n_replicas):
        """
        Produce replica indexes over which to collect.
        """
        return [{"replica_index": i} for i in range(n_replicas)]

    def produce_trval_replica_indices(
        self, n_replicas, use_same_trval_split_per_replica=False, trval_index_default=1
    ):
        """
        Produce replica and training validation split indexes over which to collect.
        """
        if use_same_trval_split_per_replica:
            return [
                {"replica_index": i, "trval_index": trval_index_default}
                for i in range(n_replicas)
            ]
        else:
            return [{"replica_index": i, "trval_index": i} for i in range(n_replicas)]

    def produce_dataset_inputs_t0_predictions(
        self, data, t0set, use_fit_t0, use_gen_t0
    ):
        """
        Produce t0 predictions for all datasets in data
        """

        if (not use_fit_t0) or (not use_gen_t0):
            raise ConfigError(
                f"use_fit_t0 or use_gen_t0 need to be set to True so that dataset_inputs_t0_predictions can be generated"
            )
        t0_predictions = []
        for dataset in data.datasets:
            t0_predictions.append(covmats.dataset_t0_predictions(dataset, t0set))
        return t0_predictions

    def parse_closure_test_pdf(self, name):
        """PDF set used to generate fakedata"""
        return self.parse_pdf(name)

    def produce_flavour_indices(self, flavour_mapping=None):
        """
        Produce flavour indices according to flavour_mapping.

        Parameters
        ----------
        flavour_mapping: list, default is None
            list of flavours names in the evolution basis
            (see e.g. validphys.convolution.FK_FLAVOURS).
            Specified by the user in the runcard.
        """

        if flavour_mapping is not None:
            indices = []
            for fl in flavour_mapping:
                if fl in FLAVOUR_TO_ID_MAPPING.keys():
                    indices.append(FLAVOUR_TO_ID_MAPPING[fl])
                else:
                    raise KeyError(f"flavour {fl} not found in FLAVOUR_TO_ID_MAPPING")
            return indices

        return None
