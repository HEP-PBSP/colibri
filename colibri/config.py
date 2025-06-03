"""
colibri.config.py

Config module of colibri

Note: several functions are taken from validphys.config
"""

import hashlib
import logging
import os
import shutil

import jax
import jax.numpy as jnp
from colibri import commondata_utils
from colibri import covmats as colibri_covmats
from colibri.constants import FLAVOUR_TO_ID_MAPPING
from colibri.core import IntegrabilitySettings, PriorSettings
from mpi4py import MPI
from reportengine.configparser import ConfigError, explicit_node
from validphys import covmats
from validphys.config import Config, Environment
from validphys.fkparser import load_fktable

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

log = logging.getLogger(__name__)


class EnvironmentError_(Exception):
    pass


class Environment(Environment):
    def __init__(
        self, replica_index=None, trval_index=0, float32=False, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.replica_index = replica_index
        self.trval_index = trval_index
        self.float32 = float32

        if self.float32:
            log.info("Using float32 precision")
            log.warning(
                "If running with ultranest, only SliceSampler is supported with float32 precision."
            )
            jax.config.update("jax_enable_x64", False)
        else:
            log.info("Using float64 precision")
            jax.config.update("jax_enable_x64", True)

    @classmethod
    def ns_dump_description(cls):
        return {
            "replica_index": "The MC replica index",
            "trval_index": "The Training/Validation split index",
            **super().ns_dump_description(),
        }

    def init_output(self):
        # Only master process creates the output folder
        if rank == 0:
            if self.output_path and self.output_path.is_dir():
                log.warning(
                    "Output folder exists: %s Overwriting contents" % self.output_path
                )
            else:
                try:
                    self.output_path.mkdir()
                except OSError as e:
                    raise EnvironmentError_(e) from e

        self.input_folder = self.output_path / "input"

        # Only master process creates the input folder and the root filter.yml file
        if rank == 0:
            self.input_folder.mkdir(exist_ok=True)
            if self.config_yml:
                try:
                    shutil.copy2(self.config_yml, self.input_folder / "runcard.yaml")
                    shutil.copy2(self.config_yml, self.output_path / "filter.yml")
                except shutil.SameFileError:
                    pass

                # Generate md5 hash of the filter.yml file
                output_filename = self.output_path / "md5"
                with open(self.output_path / "filter.yml", "rb") as f:
                    hash_md5 = hashlib.md5(f.read()).hexdigest()
                with open(output_filename, "w") as g:
                    g.write(hash_md5)

                log.info(f"md5 {hash_md5} stored in {output_filename}")

        # only master process creates the figures and tables folders
        if rank == 0:
            self.figure_folder = self.output_path / "figures"
            self.figure_folder.mkdir(exist_ok=True)

            self.table_folder = self.output_path / "tables"
            self.table_folder.mkdir(exist_ok=True)


class colibriConfig(Config):
    """
    Config class inherits from validphys
    Config class
    """

    def produce_FIT_XGRID(self, data=None, posdatasets=None):
        """
        Produces the xgrid for the fit from the union of all xgrids

        Parameters
        ----------
        data: validphys.core.DataGroupSpec
            The data object containing all datasets

        posdatasets: validphys.core.PositivitySetSpec

        Returns
        -------
        FIT_XGRID: np.array
            array from the set defined as the union of all xgrids
        """

        # compute union of all xgrids
        xgrid_points = set()
        if data is not None:
            for ds in data.datasets:

                for fkspec in ds.fkspecs:
                    fk = load_fktable(fkspec)

                    # add fktable xgrid to a set
                    xgrid_points.update(fk.xgrid)

        # repeat the same for the positivity datasets if they are defined
        if posdatasets is not None:
            for posds in posdatasets:
                for fkspec in posds.fkspecs:
                    fk = load_fktable(fkspec)

                    # add fktable xgrid to a set
                    xgrid_points.update(fk.xgrid)

        xgrid = jnp.array(sorted(xgrid_points))
        log.info(
            f"Fitting x-grid consists of {len(xgrid)} points, ranging from {xgrid[0]} to {xgrid[-1]}."
        )
        return xgrid

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
            "sampler_plot",
            "popstepsampler",
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

        return ns_settings

    def parse_positivity_penalty_settings(self, settings):
        """
        Parses the positivity_penalty_settings namespace from the runcard,
        and ensures the choice of settings is valid.
        """
        # Begin by checking that the user-supplied keys are known; warn the user otherwise.
        known_keys = {
            "alpha",
            "lambda_positivity",
            "positivity_penalty",
        }

        kdiff = settings.keys() - known_keys
        for k in kdiff:
            log.warning(
                ConfigError(
                    f"Key '{k}' in positivity_penalty_settings not known.",
                    k,
                    known_keys,
                )
            )

        # Now construct the positivity_penalty_settings dictionary, checking the parameter
        # combinations are valid
        positivity_penalty_settings = {}

        # Set the positivity penalty parameters
        positivity_penalty_settings["positivity_penalty"] = settings.get(
            "positivity_penalty", False
        )
        positivity_penalty_settings["alpha"] = settings.get("alpha", 1e-7)
        positivity_penalty_settings["lambda_positivity"] = settings.get(
            "lambda_positivity", 3000
        )

        return positivity_penalty_settings

    def parse_integrability_settings(self, settings):
        """
        Parses the integrability settings defined in the runcard
        into an IntegrabilitySettings dataclass.
        """

        known_keys = {
            "integrability",
            "integrability_specs",
        }

        kdiff = settings.keys() - known_keys
        for k in kdiff:
            # raise error if key not in known keys as otherwise IntegrabilitySettigs would
            # be passed an uknown key
            raise (
                ConfigError(
                    f"Key '{k}' in integrability_settings not known.",
                    k,
                    known_keys,
                )
            )

        integrability_settings = {}

        integrability_settings["integrability"] = settings.get("integrability", False)
        integrability_settings["integrability_specs"] = settings.get(
            "integrability_specs", {}
        )

        # assign default values
        integrability_settings["integrability_specs"].setdefault(
            "lambda_integrability", 100
        )
        integrability_settings["integrability_specs"].setdefault(
            "evolution_flavours", [9, 10]
        )  # T3 and T8 as default
        integrability_settings["integrability_specs"].setdefault(
            "integrability_xgrid", [2.00000000e-07]
        )  # last point of XGRID as default

        if integrability_settings["integrability_specs"]["evolution_flavours"] != [
            9,
            10,
        ]:  # only process if not default

            ev_fls = []

            for ev_fl in integrability_settings["integrability_specs"][
                "evolution_flavours"
            ]:
                if ev_fl not in FLAVOUR_TO_ID_MAPPING.keys():
                    raise (
                        ConfigError(
                            f"evolution_flavours ids can only be taken from  {FLAVOUR_TO_ID_MAPPING.keys()}"
                        )
                    )
                ev_fls.append(FLAVOUR_TO_ID_MAPPING[ev_fl])

            # convert strings to numeric indexes
            integrability_settings["integrability_specs"]["evolution_flavours"] = ev_fls

        return IntegrabilitySettings(**integrability_settings)

    def parse_prior_settings(self, settings):
        """
        Parses the prior_settings namespace from the runcard,
        into the core.PriorSettings dataclass.
        """
        # Begin by checking that the user-supplied keys are known; warn the user otherwise.
        known_keys = {
            "prior_distribution",
            "prior_distribution_specs",
        }

        kdiff = settings.keys() - known_keys
        for k in kdiff:
            log.warning(
                ConfigError(f"Key '{k}' in prior_settings not known.", k, known_keys)
            )

        # Now construct the prior_settings dictionary, checking the parameter combinations are valid
        prior_settings = {}

        # Set the prior distribution
        prior_settings["prior_distribution"] = settings.get(
            "prior_distribution", "uniform_parameter_prior"
        )

        # Set the prior distribution specs
        # log warning if the user has not provided the prior_distribution_specs and the prior distribution is uniform
        if (settings["prior_distribution"] == "uniform_parameter_prior") and (
            "prior_distribution_specs" not in settings
        ):
            log.warning(
                ConfigError(
                    "prior_distribution_specs not found in prior_settings. Using default [-1,1] values for uniform_parameter_prior.",
                )
            )

        # raise error if prior_distribution_specs is not provided for prior_from_gauss_posterior
        if (settings["prior_distribution"] == "prior_from_gauss_posterior") and (
            "prior_distribution_specs" not in settings
        ):
            raise ConfigError(
                "prior_distribution_specs not found in prior_settings. Please provide prior_distribution_specs for prior_from_gauss_posterior."
            )

        prior_settings["prior_distribution_specs"] = settings.get(
            "prior_distribution_specs", {"max_val": 1.0, "min_val": -1.0}
        )

        return PriorSettings(**prior_settings)

    def parse_analytic_settings(
        self,
        settings,
    ):
        """For an analytic fit, parses the analytic_settings namespace from the runcard,
        and ensures the choice of settings is valid.
        """

        # Begin by checking that the user-supplied keys are known; warn the user otherwise.
        known_keys = {
            "n_posterior_samples",
            "sampling_seed",
            "full_sample_size",
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
        analytic_settings["n_posterior_samples"] = settings.get(
            "n_posterior_samples", 100
        )

        # Set the full sample size
        analytic_settings["full_sample_size"] = settings.get("full_sample_size", 1000)

        return analytic_settings

    def produce_vectorized(self, ns_settings):
        """Returns True if the fit is vectorized, False otherwise.
        This is required for the predictions functions, which do not take ns_settings as an argument.
        """
        return ns_settings["ReactiveNS_settings"]["vectorized"]

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
            raise ConfigError(
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

    def parse_closure_test_pdf(self, name):
        """PDF set used to generate fakedata"""
        if name == "colibri_model":
            return name
        else:
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

    def produce_pdf_model(self):
        """
        Returns None as the pdf_model is not used in the colibri module.
        """
        return None

    def parse_closure_test_colibri_model_pdf(self, settings):
        """
        Validates that required keys are present and returns the full settings dictionary.
        Requires: 'model' and 'parameters'.
        Other keys (e.g. 'fitted_flavours') are allowed and passed through.
        """
        required_keys = {"model", "parameters"}

        missing_keys = required_keys - settings.keys()
        if missing_keys:
            raise KeyError(
                f"Missing required key(s) in closure_test_model_settings: {', '.join(missing_keys)}"
            )

        # Return a full copy of the settings dictionary (assuming it’s valid)
        return dict(settings)
