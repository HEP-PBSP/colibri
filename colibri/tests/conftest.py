"""
Module containing standard pytest data configurations for testing purposes.
"""

from unittest.mock import Mock

import jax
import jax.numpy as jnp
import numpy as np

from colibri.pdf_model import PDFModel

CONFIG_YML_PATH = "test_runcards/test_config.yaml"

"""
Intrinsic charm theory used in the tests.
"""
TEST_THEORYID = 40000000

"""
Default cuts to be used when testing.
"""
TEST_USECUTS = "internal"


TEST_DATASET = {
    "dataset_input": {"dataset": "NMC_NC_NOTFIXED_P_EM-SIGMARED", "variant": "legacy"},
    "theoryid": TEST_THEORYID,
    "use_cuts": TEST_USECUTS,
}

"""
This should contain the exact same info as TEST_DATASET, but with the use of
the "dataset_inputs" key instead of "dataset_input"
"""
TEST_DATASETS = {
    "dataset_inputs": [
        {"dataset": "NMC_NC_NOTFIXED_P_EM-SIGMARED", "variant": "legacy"}
    ],
    "theoryid": TEST_THEORYID,
    "use_cuts": TEST_USECUTS,
}

TEST_DATASET_HAD = {
    "dataset_input": {"dataset": "ATLAS_DY_7TEV_46FB_CC", "variant": "legacy"},
    "theoryid": TEST_THEORYID,
    "use_cuts": TEST_USECUTS,
}

"""
This should contain the exact same info as TEST_DATASET_HAD, but with the use of
the "dataset_inputs" key instead of "dataset_input"
"""
TEST_DATASETS_HAD = {
    "dataset_inputs": [{"dataset": "ATLAS_DY_7TEV_46FB_CC", "variant": "legacy"}],
    "theoryid": TEST_THEORYID,
    "use_cuts": TEST_USECUTS,
}

"""
Mixed DIS and HAD dataset for testing purposes.
"""
TEST_DATASETS_DIS_HAD = {
    "dataset_inputs": [
        {"dataset": "HERA_NC_318GEV_EP-SIGMARED", "variant": "legacy"},
        {"dataset": "ATLAS_DY_7TEV_46FB_CC", "variant": "legacy"},
    ],
    "theoryid": TEST_THEORYID,
    "use_cuts": TEST_USECUTS,
}

"""
Positivity dataset for testing purposes.
"""
TEST_POS_DATASET = {
    "positivity": {
        "posdatasets": [
            {
                "dataset": "NNPDF_POS_2P24GEV_F2U",
                "maxlambda": 1e6,
            }
        ]
    }
}

TEST_SINGLE_POS_DATASET = {
    "posdataset": {
        "dataset": "NNPDF_POS_2P24GEV_F2U",
        "maxlambda": 1e6,
    }
}

TEST_SINGLE_POS_DATASET_HAD = {
    "posdataset": {
        "dataset": "NNPDF_POS_2P24GEV_DYD",
        "maxlambda": 1e6,
    }
}


T0_PDFSET = {"t0pdfset": "NNPDF40_nnlo_as_01180"}

CLOSURE_TEST_PDFSET = {"closure_test_pdf": "NNPDF40_nnlo_as_01180"}

TRVAL_INDEX = {"trval_index": 1}
REPLICA_INDEX = {"replica_index": 1}

TEST_PDFSET = "NNPDF40_nnlo_as_01180"

PSEUDODATA_SEED = 123456


TEST_FULL_DIS_DATASET = {
    "dataset_inputs": [
        {"dataset": "NMC_NC_NOTFIXED_P_EM-SIGMARED", "variant": "legacy"},
        {"dataset": "SLAC_NC_NOTFIXED_P_DW_EM-F2", "variant": "legacy"},
        {"dataset": "SLAC_NC_NOTFIXED_D_DW_EM-F2", "variant": "legacy"},
        {"dataset": "BCDMS_NC_NOTFIXED_P_DW_EM-F2", "variant": "legacy"},
        {"dataset": "BCDMS_NC_NOTFIXED_D_DW_EM-F2", "variant": "legacy"},
        {"dataset": "CHORUS_CC_NOTFIXED_PB_DW_NU-SIGMARED", "variant": "legacy"},
        {"dataset": "CHORUS_CC_NOTFIXED_PB_DW_NB-SIGMARED", "variant": "legacy"},
        {
            "dataset": "NUTEV_CC_NOTFIXED_FE_DW_NU-SIGMARED",
            "variant": "legacy",
            "cfac": ["MAS"],
        },
        {
            "dataset": "NUTEV_CC_NOTFIXED_FE_DW_NB-SIGMARED",
            "variant": "legacy",
            "cfac": ["MAS"],
        },
        {"dataset": "HERA_NC_318GEV_EM-SIGMARED", "variant": "legacy"},
        {"dataset": "HERA_NC_251GEV_EP-SIGMARED", "variant": "legacy"},
        {"dataset": "HERA_NC_300GEV_EP-SIGMARED", "variant": "legacy"},
        {"dataset": "HERA_NC_318GEV_EP-SIGMARED", "variant": "legacy"},
        {"dataset": "HERA_NC_225GEV_EP-SIGMARED", "variant": "legacy"},
        {"dataset": "HERA_CC_318GEV_EP-SIGMARED", "variant": "legacy"},
        {"dataset": "HERA_CC_318GEV_EM-SIGMARED", "variant": "legacy"},
        {"dataset": "HERA_NC_318GEV_EAVG_BOTTOM-SIGMARED", "variant": "legacy"},
        {"dataset": "HERA_NC_318GEV_EAVG_CHARM-SIGMARED", "variant": "legacy"},
    ],
    "theoryid": TEST_THEORYID,
    "use_cuts": TEST_USECUTS,
}


TEST_FULL_HAD_DATASET = {
    "dataset_inputs": [
        # Hadronic
        {"dataset": "DYE866_Z0_800GEV_DW_RATIO_PDXSECRATIO", "variant": "legacy"},
        {"dataset": "DYE866_Z0_800GEV_PXSEC", "variant": "legacy"},
        {"dataset": "DYE605_Z0_38P8GEV_DW_PXSEC", "variant": "legacy"},
        {
            "dataset": "DYE906_Z0_120GEV_DW_PDXSECRATIO",
            "variant": "legacy",
            "cfac": ["ACC"],
        },
        {"dataset": "CDF_Z0_1P96TEV_ZRAP", "variant": "legacy"},
        {"dataset": "D0_Z0_1P96TEV_ZRAP", "variant": "legacy"},
        {"dataset": "D0_WPWM_1P96TEV_ASY", "variant": "legacy"},
        {"dataset": "ATLAS_DY_7TEV_36PB_ETA", "variant": "legacy"},
        {"dataset": "ATLAS_Z0_7TEV_49FB_HIMASS", "variant": "legacy"},
        {"dataset": "ATLAS_Z0_7TEV_LOMASS_M", "variant": "legacy"},
        {"dataset": "ATLAS_DY_7TEV_46FB_CC", "variant": "legacy"},
        {"dataset": "ATLAS_Z0_7TEV_46FB_CF-Y", "variant": "legacy"},
        {"dataset": "ATLAS_Z0_8TEV_HIMASS_M-Y", "variant": "legacy"},
        {"dataset": "ATLAS_Z0_8TEV_LOWMASS_M-Y", "variant": "legacy"},
        {"dataset": "ATLAS_DY_13TEV_TOT", "variant": "legacy", "cfac": ["NRM"]},
        {"dataset": "ATLAS_WJ_8TEV_WP-PT", "variant": "legacy"},
        {"dataset": "ATLAS_WJ_8TEV_WM-PT", "variant": "legacy"},
        {"dataset": "ATLAS_Z0J_8TEV_PT-M", "variant": "legacy_10"},
        {"dataset": "ATLAS_Z0J_8TEV_PT-Y", "variant": "legacy_10"},
        {"dataset": "ATLAS_TTBAR_7TEV_TOT_X-SEC", "variant": "legacy"},
        {"dataset": "ATLAS_TTBAR_8TEV_TOT_X-SEC", "variant": "legacy"},
        {"dataset": "ATLAS_TTBAR_13TEV_TOT_X-SEC", "variant": "legacy"},
        {"dataset": "ATLAS_TTBAR_8TEV_LJ_DIF_YT-NORM", "variant": "legacy"},
        {"dataset": "ATLAS_TTBAR_8TEV_LJ_DIF_YTTBAR-NORM", "variant": "legacy"},
        {"dataset": "ATLAS_TTBAR_8TEV_2L_DIF_YTTBAR-NORM", "variant": "legacy"},
        {"dataset": "ATLAS_PH_13TEV_XSEC", "variant": "legacy", "cfac": ["EWK"]},
        {"dataset": "ATLAS_SINGLETOP_7TEV_TCHANNEL-XSEC", "variant": "legacy"},
        {"dataset": "ATLAS_SINGLETOP_13TEV_TCHANNEL-XSEC", "variant": "legacy"},
        {"dataset": "ATLAS_SINGLETOP_7TEV_T-Y-NORM", "variant": "legacy"},
        {"dataset": "ATLAS_SINGLETOP_7TEV_TBAR-Y-NORM", "variant": "legacy"},
        {"dataset": "ATLAS_SINGLETOP_8TEV_T-RAP-NORM", "variant": "legacy"},
        {"dataset": "ATLAS_SINGLETOP_8TEV_TBAR-RAP-NORM", "variant": "legacy"},
        {"dataset": "CMS_WPWM_7TEV_ELECTRON_ASY"},
        {"dataset": "CMS_WPWM_7TEV_MUON_ASY", "variant": "legacy"},
        {"dataset": "CMS_Z0_7TEV_DIMUON_2D"},
        {"dataset": "CMS_WPWM_8TEV_MUON_Y", "variant": "legacy"},
        {"dataset": "CMS_Z0J_8TEV_PT-Y", "cfac": ["NRM"], "variant": "legacy_10"},
        {"dataset": "CMS_TTBAR_7TEV_TOT_X-SEC", "variant": "legacy"},
        {"dataset": "CMS_TTBAR_8TEV_TOT_X-SEC", "variant": "legacy"},
        {"dataset": "CMS_TTBAR_13TEV_TOT_X-SEC", "variant": "legacy"},
        {"dataset": "CMS_TTBAR_8TEV_LJ_DIF_YTTBAR-NORM", "variant": "legacy"},
        {"dataset": "CMS_TTBAR_5TEV_TOT_X-SEC", "variant": "legacy"},
        {"dataset": "CMS_TTBAR_8TEV_2L_DIF_MTTBAR-YT-NORM", "variant": "legacy"},
        {"dataset": "CMS_TTBAR_13TEV_2L_DIF_YT", "variant": "legacy"},
        {"dataset": "CMS_TTBAR_13TEV_LJ_2016_DIF_YTTBAR", "variant": "legacy"},
        {"dataset": "CMS_SINGLETOP_7TEV_TCHANNEL-XSEC", "variant": "legacy"},
        {"dataset": "CMS_SINGLETOP_8TEV_TCHANNEL-XSEC", "variant": "legacy"},
        {"dataset": "CMS_SINGLETOP_13TEV_TCHANNEL-XSEC", "variant": "legacy"},
        {"dataset": "LHCB_Z0_7TEV_DIELECTRON_Y"},
        {"dataset": "LHCB_Z0_8TEV_DIELECTRON_Y"},
        {"dataset": "LHCB_DY_7TEV_MUON_Y", "cfac": ["NRM"]},
        {"dataset": "LHCB_DY_8TEV_MUON_Y", "cfac": ["NRM"]},
        {"dataset": "LHCB_Z0_13TEV_DIMUON-Y"},
        {"dataset": "LHCB_Z0_13TEV_DIELECTRON-Y"},
    ],
    "theoryid": TEST_THEORYID,
    "use_cuts": TEST_USECUTS,
}


TEST_FULL_GLOBAL_DATASET = {
    "dataset_inputs": [
        # DIS
        {"dataset": "NMC_NC_NOTFIXED_P_EM-SIGMARED", "variant": "legacy"},
        {"dataset": "SLAC_NC_NOTFIXED_P_DW_EM-F2", "variant": "legacy"},
        {"dataset": "SLAC_NC_NOTFIXED_D_DW_EM-F2", "variant": "legacy"},
        {"dataset": "BCDMS_NC_NOTFIXED_P_DW_EM-F2", "variant": "legacy"},
        {"dataset": "BCDMS_NC_NOTFIXED_D_DW_EM-F2", "variant": "legacy"},
        {"dataset": "CHORUS_CC_NOTFIXED_PB_DW_NU-SIGMARED", "variant": "legacy"},
        {"dataset": "CHORUS_CC_NOTFIXED_PB_DW_NB-SIGMARED", "variant": "legacy"},
        {
            "dataset": "NUTEV_CC_NOTFIXED_FE_DW_NU-SIGMARED",
            "variant": "legacy",
            "cfac": ["MAS"],
        },
        {
            "dataset": "NUTEV_CC_NOTFIXED_FE_DW_NB-SIGMARED",
            "variant": "legacy",
            "cfac": ["MAS"],
        },
        {"dataset": "HERA_NC_318GEV_EM-SIGMARED", "variant": "legacy"},
        {"dataset": "HERA_NC_251GEV_EP-SIGMARED", "variant": "legacy"},
        {"dataset": "HERA_NC_300GEV_EP-SIGMARED", "variant": "legacy"},
        {"dataset": "HERA_NC_318GEV_EP-SIGMARED", "variant": "legacy"},
        {"dataset": "HERA_NC_225GEV_EP-SIGMARED", "variant": "legacy"},
        {"dataset": "HERA_CC_318GEV_EP-SIGMARED", "variant": "legacy"},
        {"dataset": "HERA_CC_318GEV_EM-SIGMARED", "variant": "legacy"},
        {"dataset": "HERA_NC_318GEV_EAVG_BOTTOM-SIGMARED", "variant": "legacy"},
        {"dataset": "HERA_NC_318GEV_EAVG_CHARM-SIGMARED", "variant": "legacy"},
        # Hadronic
        {"dataset": "DYE866_Z0_800GEV_DW_RATIO_PDXSECRATIO", "variant": "legacy"},
        {"dataset": "DYE866_Z0_800GEV_PXSEC", "variant": "legacy"},
        {"dataset": "DYE605_Z0_38P8GEV_DW_PXSEC", "variant": "legacy"},
        {
            "dataset": "DYE906_Z0_120GEV_DW_PDXSECRATIO",
            "variant": "legacy",
            "cfac": ["ACC"],
        },
        {"dataset": "CDF_Z0_1P96TEV_ZRAP", "variant": "legacy"},
        {"dataset": "D0_Z0_1P96TEV_ZRAP", "variant": "legacy"},
        {"dataset": "D0_WPWM_1P96TEV_ASY", "variant": "legacy"},
        {"dataset": "ATLAS_DY_7TEV_36PB_ETA", "variant": "legacy"},
        {"dataset": "ATLAS_Z0_7TEV_49FB_HIMASS", "variant": "legacy"},
        {"dataset": "ATLAS_Z0_7TEV_LOMASS_M", "variant": "legacy"},
        {"dataset": "ATLAS_DY_7TEV_46FB_CC", "variant": "legacy"},
        {"dataset": "ATLAS_Z0_7TEV_46FB_CF-Y", "variant": "legacy"},
        {"dataset": "ATLAS_Z0_8TEV_HIMASS_M-Y", "variant": "legacy"},
        {"dataset": "ATLAS_Z0_8TEV_LOWMASS_M-Y", "variant": "legacy"},
        {"dataset": "ATLAS_DY_13TEV_TOT", "variant": "legacy", "cfac": ["NRM"]},
        {"dataset": "ATLAS_WJ_8TEV_WP-PT", "variant": "legacy"},
        {"dataset": "ATLAS_WJ_8TEV_WM-PT", "variant": "legacy"},
        {"dataset": "ATLAS_Z0J_8TEV_PT-M", "variant": "legacy_10"},
        {"dataset": "ATLAS_Z0J_8TEV_PT-Y", "variant": "legacy_10"},
        {"dataset": "ATLAS_TTBAR_7TEV_TOT_X-SEC", "variant": "legacy"},
        {"dataset": "ATLAS_TTBAR_8TEV_TOT_X-SEC", "variant": "legacy"},
        {"dataset": "ATLAS_TTBAR_13TEV_TOT_X-SEC", "variant": "legacy"},
        {"dataset": "ATLAS_TTBAR_8TEV_LJ_DIF_YT-NORM", "variant": "legacy"},
        {"dataset": "ATLAS_TTBAR_8TEV_LJ_DIF_YTTBAR-NORM", "variant": "legacy"},
        {"dataset": "ATLAS_TTBAR_8TEV_2L_DIF_YTTBAR-NORM", "variant": "legacy"},
        {"dataset": "ATLAS_PH_13TEV_XSEC", "variant": "legacy", "cfac": ["EWK"]},
        {"dataset": "ATLAS_SINGLETOP_7TEV_TCHANNEL-XSEC", "variant": "legacy"},
        {"dataset": "ATLAS_SINGLETOP_13TEV_TCHANNEL-XSEC", "variant": "legacy"},
        {"dataset": "ATLAS_SINGLETOP_7TEV_T-Y-NORM", "variant": "legacy"},
        {"dataset": "ATLAS_SINGLETOP_7TEV_TBAR-Y-NORM", "variant": "legacy"},
        {"dataset": "ATLAS_SINGLETOP_8TEV_T-RAP-NORM", "variant": "legacy"},
        {"dataset": "ATLAS_SINGLETOP_8TEV_TBAR-RAP-NORM", "variant": "legacy"},
        {"dataset": "CMS_WPWM_7TEV_ELECTRON_ASY"},
        {"dataset": "CMS_WPWM_7TEV_MUON_ASY", "variant": "legacy"},
        {"dataset": "CMS_Z0_7TEV_DIMUON_2D"},
        {"dataset": "CMS_WPWM_8TEV_MUON_Y", "variant": "legacy"},
        {"dataset": "CMS_Z0J_8TEV_PT-Y", "cfac": ["NRM"], "variant": "legacy_10"},
        {"dataset": "CMS_TTBAR_7TEV_TOT_X-SEC", "variant": "legacy"},
        {"dataset": "CMS_TTBAR_8TEV_TOT_X-SEC", "variant": "legacy"},
        {"dataset": "CMS_TTBAR_13TEV_TOT_X-SEC", "variant": "legacy"},
        {"dataset": "CMS_TTBAR_8TEV_LJ_DIF_YTTBAR-NORM", "variant": "legacy"},
        {"dataset": "CMS_TTBAR_5TEV_TOT_X-SEC", "variant": "legacy"},
        {"dataset": "CMS_TTBAR_8TEV_2L_DIF_MTTBAR-YT-NORM", "variant": "legacy"},
        {"dataset": "CMS_TTBAR_13TEV_2L_DIF_YT", "variant": "legacy"},
        {"dataset": "CMS_TTBAR_13TEV_LJ_2016_DIF_YTTBAR", "variant": "legacy"},
        {"dataset": "CMS_SINGLETOP_7TEV_TCHANNEL-XSEC", "variant": "legacy"},
        {"dataset": "CMS_SINGLETOP_8TEV_TCHANNEL-XSEC", "variant": "legacy"},
        {"dataset": "CMS_SINGLETOP_13TEV_TCHANNEL-XSEC", "variant": "legacy"},
        {"dataset": "LHCB_Z0_7TEV_DIELECTRON_Y"},
        {"dataset": "LHCB_Z0_8TEV_DIELECTRON_Y"},
        {"dataset": "LHCB_DY_7TEV_MUON_Y", "cfac": ["NRM"]},
        {"dataset": "LHCB_DY_8TEV_MUON_Y", "cfac": ["NRM"]},
        {"dataset": "LHCB_Z0_13TEV_DIMUON-Y"},
        {"dataset": "LHCB_Z0_13TEV_DIELECTRON-Y"},
    ],
    "theoryid": TEST_THEORYID,
    "use_cuts": TEST_USECUTS,
}


TEST_FULL_POS_DATASET = {
    "positivity": {
        "posdatasets": [
            {
                "dataset": "NNPDF_POS_2P24GEV_F2U",
                "maxlambda": 1e6,
            },  # Positivity Lagrange Multiplier
            {"dataset": "NNPDF_POS_2P24GEV_F2D", "maxlambda": 1e6},
            {"dataset": "NNPDF_POS_2P24GEV_F2S", "maxlambda": 1e6},
            {"dataset": "NNPDF_POS_2P24GEV_FLL-19PTS", "maxlambda": 1e6},
            {"dataset": "NNPDF_POS_2P24GEV_DYU", "maxlambda": 1e10},
            {"dataset": "NNPDF_POS_2P24GEV_DYD", "maxlambda": 1e10},
            {"dataset": "NNPDF_POS_2P24GEV_DYS", "maxlambda": 1e10},
            {"dataset": "NNPDF_POS_2P24GEV_F2C-17PTS", "maxlambda": 1e6},
            {
                "dataset": "NNPDF_POS_2P24GEV_XUQ",
                "maxlambda": 1e6,
            },  # Positivity of MSbar PDFs
            {"dataset": "NNPDF_POS_2P24GEV_XUB", "maxlambda": 1e6},
            {"dataset": "NNPDF_POS_2P24GEV_XDQ", "maxlambda": 1e6},
            {"dataset": "NNPDF_POS_2P24GEV_XDB", "maxlambda": 1e6},
            {"dataset": "NNPDF_POS_2P24GEV_XSQ", "maxlambda": 1e6},
            {"dataset": "NNPDF_POS_2P24GEV_XSB", "maxlambda": 1e6},
            {"dataset": "NNPDF_POS_2P24GEV_XGL", "maxlambda": 1e6},
        ]
    },
    "theoryid": TEST_THEORYID,
    "use_cuts": TEST_USECUTS,
}


"""
Toy PDF model to be used to test the pdf_model module.
"""
N_PARAMS = 10


class TestPDFModel(PDFModel):

    def __init__(self, n_parameters):
        self.n_parameters = n_parameters

    @property
    def param_names(self):
        return [f"w_{i+1}" for i in range(self.n_parameters)]

    def grid_values_func(self, xgrid):
        """
        This function should produce a grid values function, which takes
        in the model parameters, and produces the PDF values on the grid xgrid.
        """

        def wmin_param(params):
            """
            Returns random array of shape (14,len(params))
            """
            return sum([param * xgrid for param in params])

        return wmin_param


"""
Mock PDF model to be used to test functions that require a PDFModel instance.
"""
MOCK_PDF_MODEL = Mock()
MOCK_PDF_MODEL.param_names = ["param1", "param2"]
MOCK_PDF_MODEL.grid_values_func = lambda xgrid: lambda params: np.ones((14, len(xgrid)))
MOCK_PDF_MODEL.pred_and_pdf_func = (
    lambda xgrid, forward_map: lambda params, fast_kernel_arrays: (
        forward_map(MOCK_PDF_MODEL.grid_values_func(xgrid)(params)),
        np.ones((14, len(xgrid))),
    )
)

TEST_XGRID = jnp.array([0.1, 0.2])
TEST_FK_ARRAYS = (jnp.array([1, 2]),)
TEST_POS_FK_ARRAYS = (jnp.array([1, 2]),)
TEST_FORWARD_MAP = lambda pdf, fk_arrays: pdf * fk_arrays[0]


"""
Mock instance of Central Covmat Index object
"""
MOCK_CENTRAL_COVMAT_INDEX = Mock()
MOCK_CENTRAL_COVMAT_INDEX.central_values = jnp.ones(2)
MOCK_CENTRAL_COVMAT_INDEX.inv_covmat = jnp.eye(2)
MOCK_CENTRAL_COVMAT_INDEX.central_values_idx = jnp.arange(2)

"""
Mock instance of Central Inverse covmat index object
"""
MOCK_CENTRAL_INV_COVMAT_INDEX = Mock()
MOCK_CENTRAL_INV_COVMAT_INDEX.central_values = jnp.ones(2)
MOCK_CENTRAL_INV_COVMAT_INDEX.inv_covmat = jnp.eye(2)
MOCK_CENTRAL_INV_COVMAT_INDEX.central_values_idx = jnp.arange(2)

"""
Mock Log likelihood class
"""


class UltraNestLogLikelihoodMock:
    def __init__(
        self,
        central_inv_covmat_index,
        pdf_model,
        fit_xgrid,
        forward_map,
        fast_kernel_arrays,
        positivity_fast_kernel_arrays,
        ns_settings,
        chi2,
        penalty_posdata,
        positivity_penalty_settings,
    ):
        """
        Mock version of UltraNestLogLikelihood class for testing purposes.

        Parameters
        ----------
        central_inv_covmat_index: commondata_utils.CentralInvCovmatIndex

        pdf_model: pdf_model.PDFModel

        fit_xgrid: np.ndarray

        forward_map: Callable

        fast_kernel_arrays: tuple

        positivity_fast_kernel_arrays: tuple

        ns_settings: dict

        chi2: Callable

        penalty_posdata: Callable

        positivity_penalty_settings: dict
        """
        self.central_values = central_inv_covmat_index.central_values
        self.inv_covmat = central_inv_covmat_index.inv_covmat
        self.pdf_model = pdf_model
        self.chi2 = chi2
        self.penalty_posdata = penalty_posdata
        self.positivity_penalty_settings = positivity_penalty_settings

        self.pred_and_pdf = pdf_model.pred_and_pdf_func(
            fit_xgrid, forward_map=forward_map
        )

        if ns_settings["ReactiveNS_settings"]["vectorized"]:
            self.pred_and_pdf = jax.vmap(
                self.pred_and_pdf, in_axes=(0, None), out_axes=(0, 0)
            )

            self.chi2 = jax.vmap(self.chi2, in_axes=(None, 0, None), out_axes=0)
            self.penalty_posdata = jax.vmap(
                self.penalty_posdata, in_axes=(0, None, None, None), out_axes=0
            )

        self.fast_kernel_arrays = fast_kernel_arrays
        self.positivity_fast_kernel_arrays = positivity_fast_kernel_arrays

    def __call__(self, params):
        """
        Mock function called by the ultranest sampler.

        Parameters
        ----------
        params: np.array
            The model parameters.
        """
        return self.log_likelihood(
            params,
            self.central_values,
            self.inv_covmat,
            self.fast_kernel_arrays,
            self.positivity_fast_kernel_arrays,
        )

    def log_likelihood(
        self,
        params,
        central_values,
        inv_covmat,
        fast_kernel_arrays,
        positivity_fast_kernel_arrays,
    ):
        predictions, pdf = self.pred_and_pdf(params, fast_kernel_arrays)
        return -0.5 * (self.chi2(central_values, predictions, inv_covmat))
