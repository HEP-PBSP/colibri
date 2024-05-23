"""
Module containing standard pytest data configurations for testing purposes.
"""

from colibri.tests.conftest import TEST_PDFSET

EXE = "wmin"

TEST_PRIOR_SETTINGS_WMIN = {
    "prior_settings": {
        "type": "uniform_parameter_prior",
        "max_val": 10.0,
        "min_val": -10.0,
    }
}

TEST_WMIN_SETTINGS_NBASIS_10 = {
    "wmin_settings": {"wminpdfset": TEST_PDFSET, "n_basis": 10}
}

TEST_WMIN_SETTINGS_NBASIS_100 = {
    "wmin_settings": {"wminpdfset": TEST_PDFSET, "n_basis": 100}
}


RUNCARD_WMIN_LIKELIHOOD_TYPE = "wmin_likelihood_type_test.yaml"
