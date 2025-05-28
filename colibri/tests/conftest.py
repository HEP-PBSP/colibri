"""
Module containing standard pytest data configurations for testing purposes.
"""

import pathlib
from unittest.mock import Mock, MagicMock

import jax
import jax.numpy as jnp
import numpy as np

from colibri.pdf_model import PDFModel
from colibri.core import PriorSettings


CONFIG_YML_PATH = "test_runcards/test_config.yaml"

TEST_THEORYID = 40_000_000
"""
Intrinsic charm theory used in the tests.
"""

TEST_USECUTS = "internal"
"""
Default cuts to be used when testing.
"""

TEST_DATASET = {
    "dataset_input": {"dataset": "NMC_NC_NOTFIXED_P_EM-SIGMARED", "variant": "legacy"},
    "theoryid": TEST_THEORYID,
    "use_cuts": TEST_USECUTS,
}

TEST_DATASETS = {
    "dataset_inputs": [
        {"dataset": "NMC_NC_NOTFIXED_P_EM-SIGMARED", "variant": "legacy"}
    ],
    "theoryid": TEST_THEORYID,
    "use_cuts": TEST_USECUTS,
}
"""
This should contain the exact same info as TEST_DATASET, but with the use of
the "dataset_inputs" key instead of "dataset_input"
"""

TEST_DATASET_HAD = {
    "dataset_input": {"dataset": "ATLAS_DY_7TEV_46FB_CC", "variant": "legacy"},
    "theoryid": TEST_THEORYID,
    "use_cuts": TEST_USECUTS,
}

TEST_DATASETS_HAD = {
    "dataset_inputs": [{"dataset": "ATLAS_DY_7TEV_46FB_CC", "variant": "legacy"}],
    "theoryid": TEST_THEORYID,
    "use_cuts": TEST_USECUTS,
}
"""
This should contain the exact same info as TEST_DATASET_HAD, but with the use of
the "dataset_inputs" key instead of "dataset_input"
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
Mixed DIS and HAD dataset for testing purposes.
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
"""
Positivity dataset for testing purposes.
"""

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


TEST_N_XGRID = 50
"""
Default number of xgrid points to be used in the tests.
"""

TEST_N_FL = 14
"""
Default number of flavours to be used in the tests.
"""

TEST_PDF_GRID = np.ones((TEST_N_FL, TEST_N_XGRID))
"""
Test PDF grid used for testing purposes.
"""


class TestPDFModel(PDFModel):
    """
    Toy PDF model to be used to test the pdf_model module.
    This is needed to test the properties and methods of the PDFModel class.
    For other purposes, we can just use the Mock class.
    """

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
            Returns random array of shape (TEST_N_FL,len(params)).
            """
            return sum([param * TEST_PDF_GRID for param in params])

        return wmin_param


MOCK_PDF_MODEL = Mock()
MOCK_PDF_MODEL.param_names = ["param1", "param2"]
MOCK_PDF_MODEL.grid_values_func = lambda xgrid: lambda params: np.sum(
    np.array([param * TEST_PDF_GRID for param in params]), axis=0
)
"""
Mock PDF model with 2 parameters and grid_values_func simple mult add operation on np.ones grid.
"""

MOCK_PDF_MODEL.pred_and_pdf_func = (
    lambda xgrid, forward_map: lambda params, fast_kernel_arrays: (
        forward_map(MOCK_PDF_MODEL.grid_values_func(xgrid)(params), fast_kernel_arrays),
        MOCK_PDF_MODEL.grid_values_func(xgrid)(params),
    )
)
"""
Mock prediction function of PDF model.
"""


TEST_XGRID = jnp.logspace(-7, 0, TEST_N_XGRID)
"""
X-grid used for testing purposes.
"""

TEST_N_DATA = 2
"""
Number of data points used in mock models for testing purposes.
"""

np.random.seed(1)
TEST_FK_ARRAYS = (np.random.rand(TEST_N_DATA, TEST_N_FL, TEST_N_XGRID),)
"""
Tuple of fast kernel arrays used for testing purposes.
This mocks a DIS fast kernel mapping the PDF grid to 2 datapoints.
"""


np.random.seed(2)
TEST_POS_FK_ARRAYS = (np.random.rand(TEST_N_DATA, TEST_N_FL, TEST_N_XGRID),)
"""
Tuple of fast kernel arrays used for testing purposes.
This mocks a POS fast kernel mapping the PDF grid to 2 datapoints.
"""


TEST_FORWARD_MAP_DIS = lambda pdf, fk_arrays: np.einsum("ijk,jk->i", fk_arrays, pdf)
"""
Mock DIS forward map function for testing purposes.
Function expects a DIS-like fast kernel array of shape (N_data, TEST_N_FL, TEST_N_XGRID) and a PDF of shape (TEST_N_FL, TEST_N_XGRID).
"""


MOCK_CENTRAL_INV_COVMAT_INDEX = Mock()
MOCK_CENTRAL_INV_COVMAT_INDEX.central_values = jnp.ones(TEST_N_DATA)
MOCK_CENTRAL_INV_COVMAT_INDEX.inv_covmat = jnp.eye(TEST_N_DATA)
MOCK_CENTRAL_INV_COVMAT_INDEX.central_values_idx = jnp.arange(TEST_N_DATA)
"""
Mock instance of Central Inverse covmat index object.
"""


MOCK_CHI2 = MagicMock(return_value=10.0)
"""
Mock chi2 function for testing purposes.
"""

MOCK_PENALTY_POSDATA = MagicMock(return_value=jnp.array([5.0]))
"""
Mock penalty_posdata function for testing purposes.
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


TEST_PRIOR_SETTINGS_UNIFORM = PriorSettings(
    **{
        "prior_distribution": "uniform_parameter_prior",
        "prior_distribution_specs": {"min_val": -1.0, "max_val": 1.0},
    }
)
"""
Uniform prior settings for testing purposes, `prior_distribution_specs` should correspond to the default specs for a uniform prior.
"""


TEST_COMMONDATA_FOLDER = pathlib.Path(__file__).with_name("test_commondata")
"""
Path to the folder containing the test commondata files.
"""


EXPECTED_XGRID = [
    2.00000000e-07,
    3.03430477e-07,
    4.60350147e-07,
    6.98420853e-07,
    1.05960950e-06,
    1.60758550e-06,
    2.43894329e-06,
    3.70022721e-06,
    5.61375772e-06,
    8.51680668e-06,
    1.29210157e-05,
    1.96025050e-05,
    2.97384954e-05,
    4.51143839e-05,
    6.84374492e-05,
    1.03811730e-04,
    1.57456056e-04,
    2.38787829e-04,
    3.62054496e-04,
    5.48779532e-04,
    8.31406884e-04,
    1.25867971e-03,
    1.90346340e-03,
    2.87386758e-03,
    4.32850064e-03,
    6.49620619e-03,
    9.69915957e-03,
    1.43750686e-02,
    2.10891867e-02,
    3.05215840e-02,
    4.34149174e-02,
    6.04800288e-02,
    8.22812213e-02,
    1.09143757e-01,
    1.41120806e-01,
    1.78025660e-01,
    2.19504127e-01,
    2.65113704e-01,
    3.14387401e-01,
    3.66875319e-01,
    4.22166775e-01,
    4.79898903e-01,
    5.39757234e-01,
    6.01472198e-01,
    6.64813948e-01,
    7.29586844e-01,
    7.95624252e-01,
    8.62783932e-01,
    9.30944081e-01,
    1.00000000e00,
]
"""
The expected XGRID used in the FK-tables and in colibri.
"""

EXPECTED_LHAPDF_XGRID = [
    1e-09,
    1.29708482343957e-09,
    1.68242903474257e-09,
    2.18225315420583e-09,
    2.83056741739819e-09,
    3.67148597892941e-09,
    4.76222862935315e-09,
    6.1770142737618e-09,
    8.01211109898438e-09,
    1.03923870607245e-08,
    1.34798064073805e-08,
    1.74844503691778e-08,
    2.26788118881103e-08,
    2.94163370300835e-08,
    3.81554746595878e-08,
    4.94908707232129e-08,
    6.41938295708371e-08,
    8.32647951986859e-08,
    1.08001422993829e-07,
    1.4008687308113e-07,
    1.81704331793772e-07,
    2.35685551545377e-07,
    3.05703512595323e-07,
    3.96522309841747e-07,
    5.1432125723657e-07,
    6.67115245136676e-07,
    8.65299922973143e-07,
    1.12235875241487e-06,
    1.45577995547683e-06,
    1.88824560514613e-06,
    2.44917352454946e-06,
    3.17671650028717e-06,
    4.12035415232797e-06,
    5.3442526575209e-06,
    6.93161897806315e-06,
    8.99034258238145e-06,
    1.16603030112258e-05,
    1.51228312288769e-05,
    1.96129529349212e-05,
    2.54352207134502e-05,
    3.29841683435992e-05,
    4.27707053972016e-05,
    5.54561248105849e-05,
    7.18958313632514e-05,
    9.31954227979614e-05,
    0.00012078236773133,
    0.000156497209466554,
    0.000202708936328495,
    0.000262459799331951,
    0.000339645244168985,
    0.000439234443000422,
    0.000567535660104533,
    0.000732507615725537,
    0.000944112105452451,
    0.00121469317686978,
    0.00155935306118224,
    0.00199627451141338,
    0.00254691493736552,
    0.00323597510213126,
    0.00409103436509565,
    0.00514175977083962,
    0.00641865096062317,
    0.00795137940306351,
    0.009766899996241,
    0.0118876139251364,
    0.0143298947643919,
    0.0171032279460271,
    0.0202100733925079,
    0.0236463971369542,
    0.0274026915728357,
    0.0314652506132444,
    0.0358174829282429,
    0.0404411060163317,
    0.0453171343973807,
    0.0504266347950069,
    0.0557512610084339,
    0.0612736019390519,
    0.0669773829498255,
    0.0728475589986517,
    0.0788703322292727,
    0.0850331197801452,
    0.0913244910278679,
    0.0977340879783772,
    0.104252538208639,
    0.110871366547237,
    0.117582909372878,
    0.124380233801599,
    0.131257062945031,
    0.138207707707289,
    0.145227005135651,
    0.152310263065985,
    0.159453210652156,
    0.166651954293987,
    0.173902938455578,
    0.181202910873333,
    0.188548891679097,
    0.195938145999193,
    0.203368159629765,
    0.210836617429103,
    0.218341384106561,
    0.225880487124065,
    0.233452101459503,
    0.241054536011681,
    0.248686221452762,
    0.256345699358723,
    0.264031612468684,
    0.271742695942783,
    0.279477769504149,
    0.287235730364833,
    0.295015546847664,
    0.302816252626866,
    0.310636941519503,
    0.318476762768082,
    0.326334916761672,
    0.334210651149156,
    0.342103257303627,
    0.350012067101685,
    0.357936449985571,
    0.365875810279643,
    0.373829584735962,
    0.381797240286494,
    0.389778271981947,
    0.397772201099286,
    0.40577857340234,
    0.413796957540671,
    0.421826943574548,
    0.429868141614175,
    0.437920180563205,
    0.44598270695699,
    0.454055383887562,
    0.462137890007651,
    0.470229918607142,
    0.478331176755675,
    0.486441384506059,
    0.494560274153348,
    0.502687589545177,
    0.510823085439086,
    0.518966526903235,
    0.527117688756998,
    0.535276355048428,
    0.543442318565661,
    0.551615380379768,
    0.559795349416641,
    0.5679820420558,
    0.576175281754088,
    0.584374898692498,
    0.59258072944444,
    0.60079261666395,
    0.609010408792398,
    0.61723395978245,
    0.625463128838069,
    0.633697780169485,
    0.641937782762089,
    0.650183010158361,
    0.658433340251944,
    0.666688655093089,
    0.674948840704708,
    0.683213786908386,
    0.691483387159697,
    0.699757538392251,
    0.708036140869916,
    0.716319098046733,
    0.724606316434025,
    0.732897705474271,
    0.741193177421404,
    0.749492647227008,
    0.757796032432224,
    0.766103253064927,
    0.774414231541921,
    0.782728892575836,
    0.791047163086478,
    0.799368972116378,
    0.807694250750291,
    0.816022932038457,
    0.824354950923382,
    0.832690244169987,
    0.841028750298844,
    0.8493704095226,
    0.857715163684985,
    0.866062956202683,
    0.874413732009721,
    0.882767437504206,
    0.891124020497459,
    0.899483430165226,
    0.907845617001021,
    0.916210532771399,
    0.924578130473112,
    0.932948364292029,
    0.941321189563734,
    0.949696562735755,
    0.958074441331298,
    0.966454783914439,
    0.974837550056705,
    0.983222700304978,
    0.991610196150662,
    1.0,
]
"""
The expected LHAPDF grid.
"""
