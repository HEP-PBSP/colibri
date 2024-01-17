"""
super_net.config.py

Config module of super_net

Author: Mark N. Costantini
Note: several functions are taken from validphys.config
Date: 11.11.2023
"""

from validphys.config import Config, Environment
from validphys import covmats

from super_net import covmats as super_net_covmats

from reportengine.configparser import explicit_node, ConfigError

from super_net import commondata_utils
from super_net.core import SuperNetDataGroupSpec

from super_net.utils import FLAVOUR_TO_ID_MAPPING

import logging
import os

log = logging.getLogger(__name__)


class Environment(Environment):
    pass


class SuperNetConfig(Config):
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
            "min_num_live_points",
            "min_ess",
            "n_posterior_samples",
            "posterior_resampling_seed",
            "ndraw_max",
            "vectorized",
            "slice_sampler",
            "slice_steps",
            "resume",
            "log_dir",
        }

        kdiff = settings.keys() - known_keys
        for k in kdiff:
            log.warning(
                ConfigError(f"Key '{k}' in ns_settings not known.", k, known_keys)
            )

        # Now construct the ns_settings dictionary, checking the parameter combinations are
        # valid
        ns_settings = {}

        # Set min_num_live_points and min_ess
        ns_settings["min_num_live_points"] = settings.get("min_num_live_points", 400)
        ns_settings["min_ess"] = settings.get("min_ess", 40)

        # Set the posterior resampling parameters
        ns_settings["n_posterior_samples"] = settings.get("n_posterior_samples", 1000)
        ns_settings["posterior_resampling_seed"] = settings.get(
            "posterior_resampling_seed", 123456
        )

        # Vectorization is switched off, by default
        ns_settings["vectorized"] = settings.get("vectorized", False)
        ns_settings["ndraw_max"] = settings.get("ndraw_max", 500)

        # Set the slice sampler parameters
        ns_settings["slice_sampler"] = settings.get("slice_sampler", False)
        ns_settings["slice_steps"] = settings.get("slice_steps", 100)

        # Fit will not resume from previous ultranest fit by default
        ns_settings["resume"] = settings.get("resume", False)

        # Set the directory where the ultranest logs will be stored; by default
        # they are stored in output_path/ultranest_logs
        ns_settings["log_dir"] = settings.get("log_dir", output_path / "ultranest_logs")

        # In the case that the fit is resuming from a previous ultranest fit, the logs
        # directory must exist
        if ns_settings["resume"]:
            if not os.path.exists(ns_settings["log_dir"]):
                raise FileNotFoundError(
                    "Could not find previous ultranest fit at "
                    + str(ns_settings["log_dir"])
                    + "."
                )

            log.info("Resuming ultranest fit from " + str(ns_settings["log_dir"]) + ".")

        # If the resume option is false, ultranest expects "overwrite" instead
        if not ns_settings["resume"]:
            ns_settings["resume"] = "overwrite"

        return ns_settings

    def produce_data(
        self,
        data_input,
        *,
        group_name="data",
    ):
        """A set of datasets where correlated systematics are taken
        into account
        """
        datasets = []
        for dsinp in data_input:
            with self.set_context(ns=self._curr_ns.new_child({"dataset_input": dsinp})):
                datasets.append(self.parse_from_(None, "dataset", write=False)[1])

        return SuperNetDataGroupSpec(
            name=group_name, datasets=datasets, dsinputs=data_input
        )

    @explicit_node
    def produce_commondata_tuple(self, closure_test_level=None):
        """
        Produces a commondata tuple node in the reportengine dag
        according to some options
        """
        if not closure_test_level:
            return commondata_utils.experimental_commondata_tuple
        elif closure_test_level == 0:
            return commondata_utils.level_0_commondata_tuple
        elif closure_test_level == 1:
            return commondata_utils.level_1_commondata_tuple
        else:
            raise ValueError("closure_test_level must be None, 0 or 1.")

    @explicit_node
    def produce_fit_covariance_matrix(self, use_fit_t0: bool = True):
        """Modifies which action is used as covariance matrix
        depending on the flag `use_fit_t0`
        """
        if use_fit_t0:
            return super_net_covmats.dataset_inputs_t0_covmat_from_systematics
        else:
            return super_net_covmats.dataset_inputs_covmat_from_systematics

    @explicit_node
    def produce_data_generation_covariance_matrix(self):
        """Produces the covariance matrix used in:
        - level 1 closure test data construction (fluctuating around the level
        0 data)
        - Monte Carlo pseudodata (fluctuating either around the level 0 data or
        level 1 data)
        """
        return super_net_covmats.dataset_inputs_covmat_from_systematics

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

    def produce_dataset_inputs_t0_predictions(self, data, t0set, use_fit_t0):
        """
        Produce t0 predictions for all datasets in data
        """

        if not use_fit_t0:
            raise ConfigError(
                f"use_fit_t0 needs to be set to True so that dataset_inputs_t0_predictions can be generated"
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
