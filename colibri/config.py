"""
colibri.config.py

Config module of colibri

Note: several functions are taken from validphys.config
"""

import hashlib
import logging
import shutil
import jax

import jax.numpy as jnp
from colibri import commondata_utils
from colibri import covmats as colibri_covmats
from colibri.constants import FLAVOUR_TO_ID_MAPPING
from colibri.core import (
    ColibriLossFunctionSpecs,
    ColibriPriorSpecs,
    ColibriSpecs,
)
from colibri.config_utils import ns_settings_parser, analytic_settings_parser

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

    def parse_colibri_specs(self, settings, output_path):
        """
        This parse rule is used to parse the colibri specs that are common to all fits
        and always needed.

        Parameters
        ----------
        settings: dict
            The colibri namespace from the runcard

        Returns
        -------
        ColibriSpecs
            A dataclass containing the colibri specifications
        """
        known_keys = {"loss_function_specs", "prior_settings", "ns_settings"}
        kdiff = settings.keys() - known_keys
        for k in kdiff:
            log.warning(ConfigError(f"Key '{k}' in colibri not known.", k, known_keys))

        # loss_function_specs namespace
        loss_function_specs_settings = settings.get("loss_function_specs", {})

        loss_function_specs = ColibriLossFunctionSpecs(
            use_fit_t0=loss_function_specs_settings.get("use_fit_t0", False),
            t0pdfset=loss_function_specs_settings.get("t0pdfset", None),
        )

        # prior_settings namespace
        prior_specs_settings = settings.get("prior_settings", {})
        prior_specs = ColibriPriorSpecs(prior_settings=prior_specs_settings)

        # Nested Sampling settings
        ns_settings_settings = settings.get("ns_settings", {})
        ns_settings = ns_settings_parser(ns_settings_settings, output_path)

        # Analytic settings
        analytic_settings_settings = settings.get("analytic_settings", {})
        analytic_settings = analytic_settings_parser(analytic_settings_settings)

        # create a colibri_specs instance
        col_spec = ColibriSpecs(
            loss_function_specs=loss_function_specs,
            prior_settings=prior_specs,
            ns_settings=ns_settings,
            analytic_settings=analytic_settings,
        )

        return col_spec

    def produce_prior_settings(self, colibri_specs):
        """
        Given the parsed colibri_specs, returns the prior settings.
        """
        return colibri_specs.prior_settings.prior_settings

    def produce_t0pdfset(self, colibri_specs):
        """
        Given the parsed colibri_specs, returns the t0pdfset.
        """
        return colibri_specs.loss_function_specs.t0pdfset

    def produce_use_fit_t0(self, colibri_specs):
        """
        Given the parsed colibri_specs, returns the use_fit_t0.
        """
        return colibri_specs.loss_function_specs.use_fit_t0

    def produce_ns_settings(
        self,
        colibri_specs,
    ):
        """
        Given the parsed colibri_specs, returns the ns_settings.
        """
        return colibri_specs.ns_settings.ns_settings

    def produce_analytic_settings(
        self,
        colibri_specs,
    ):
        """
        Given the parsed colibri_specs, returns the analytic_settings.
        """
        return colibri_specs.analytic_settings.analytic_settings

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

    def produce_pdf_model(self):
        """
        Returns None as the pdf_model is not used in the colibri module.
        """
        return None
