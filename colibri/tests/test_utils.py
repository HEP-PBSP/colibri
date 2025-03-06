"""
Module for testing the utils module.
"""

import os
import pathlib
import shutil
from numpy.testing import assert_allclose
import pytest
from unittest.mock import patch, mock_open

import jax
import jax.numpy as jnp
import numpy as np
import pandas
from colibri.api import API as cAPI
from colibri.tests.conftest import (
    MOCK_CENTRAL_INV_COVMAT_INDEX,
    MOCK_PDF_MODEL,
    TEST_DATASET_HAD,
    TEST_DATASET,
)
from colibri.utils import (
    t0_pdf_grid,
    closure_test_pdf_grid,
    resample_from_ns_posterior,
    cast_to_numpy,
    get_fit_path,
    get_full_posterior,
    get_pdf_model,
    likelihood_float_type,
    mask_fktable_array,
    mask_luminosity_mapping,
    compute_determinants_of_principal_minors,
    closest_indices,
)
from validphys.fkparser import load_fktable
import validphys
from validphys import convolution

SIMPLE_WMIN_FIT = "wmin_bayes_dis"


def test_t0_pdf_grid():
    """
    Test the t0_pdf_grid function.

    Verifies:
    - Type of "t0pdfset" is validphys.core.PDF
    - Output type is a jnp.array.
    - The output shape is (N_rep, N_fl, N_x)
    """

    # mock a valid PDF set
    inp = {"t0pdfset": "NNPDF40_nlo_as_01180"}
    t0pdfset = cAPI.t0pdfset(**inp)

    # Check 1: t0pdfset is an instance of validphys.core.PDF
    assert isinstance(t0pdfset, validphys.core.PDF)

    # define a test array
    FIT_XGRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # call the function
    t0_grid = t0_pdf_grid(t0pdfset, FIT_XGRID, Q0=1.65)

    # Check 2: type of the output is a jnp.array
    assert isinstance(t0_grid, jnp.ndarray)

    # Check 3: shape of the output
    N_rep = t0pdfset.get_members()  #   number of replicas
    N_fl = len(convolution.FK_FLAVOURS)  # number of flavours

    print(f"FK_FLAVOURS: {convolution.FK_FLAVOURS}")

    assert t0_grid.shape == (N_rep, N_fl, len(FIT_XGRID))


def test_closure_test_pdf_grid():
    """
    Test the closure_test_pdf_grid function.

    Verifies:
    - Type of "closure_test_pdf" is validphys.core.PDF
    - Output type is a jnp.array.
    - The output shape is (N_rep, N_fl, N_x)
    """
    # mock a valid PDF set
    inp = {"closure_test_pdf": "NNPDF40_nlo_as_01180"}
    cltest_pdf_set = cAPI.closure_test_pdf(**inp)

    # Check 1: closure_test_pdf is an instance of validphys.core.PDF
    assert isinstance(cltest_pdf_set, validphys.core.PDF)

    # define a test array
    FIT_XGRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # call the function
    grid = closure_test_pdf_grid(cltest_pdf_set, FIT_XGRID, Q0=1.65)

    # Check 2: type of the output is a jnp.array
    assert isinstance(grid, jnp.ndarray)

    # Check 3: shape of the output
    N_rep = cltest_pdf_set.get_members()  #   number of replicas
    N_fl = len(convolution.FK_FLAVOURS)  # number of flavours

    print(f"FK_FLAVOURS: {convolution.FK_FLAVOURS}")

    assert grid.shape == (N_rep, N_fl, len(FIT_XGRID))


def test_resample_from_ns_posterior():
    """
    Test the resample_from_ns_posterior function.
    Verifies:
    - Output type is a JAX DeviceArray.
    - Output size matches n_posterior_samples and is smaller than or equal to the input sample size.
    - All elements in the output belong to the original sample.
    - There are no duplicate elements in the output.
    - If n_posterior_samples equals the input size, the output is identical to the input.
    """

    # Create a sample to test the function
    samples = jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    n_posterior_samples = 3
    posterior_resampling_seed = 42

    # Call the function
    resampled_samples = resample_from_ns_posterior(
        samples,
        n_posterior_samples=n_posterior_samples,
        posterior_resampling_seed=posterior_resampling_seed,
    )

    # Check 1: Output type
    assert isinstance(resampled_samples, jnp.ndarray)

    # Check 2: Output size
    assert len(resampled_samples) == n_posterior_samples
    assert len(resampled_samples) <= len(samples)

    # Check 3: All elements in output belong to the original samples
    assert np.all(np.isin(resampled_samples, samples))

    # Check 4: No duplicates
    assert len(resampled_samples) == len(jnp.unique(resampled_samples))

    # Case 2: n_posterior_samples equals the size of the input samples
    n_posterior_samples = len(samples)
    resampled_samples_full = resample_from_ns_posterior(
        samples,
        n_posterior_samples=n_posterior_samples,
        posterior_resampling_seed=posterior_resampling_seed,
    )

    # Check 5: Output is identical to the input when sizes match
    assert jnp.array_equal(jnp.sort(resampled_samples_full), jnp.sort(samples))


def test_cast_to_numpy():
    """
    test the cast_to_numpy function
    """

    @cast_to_numpy
    @jax.jit
    def test_func(x):
        return x

    x = jax.numpy.array([1, 2, 3])

    assert type(test_func(x)) == np.ndarray


def test_get_path_fit():
    """
    Tests the get_fit_path function.
    Copies the wmin_bayes_dis directory to $CONDA_PREFIX/share/colibri/results
    and checks if the function returns the correct path.
    Finally, it removes the copied directory.
    """

    conda_prefix = os.getenv("CONDA_PREFIX")

    destination_dir = pathlib.Path(conda_prefix) / "share" / "colibri" / "results"
    source_dir = pathlib.Path("colibri/tests/regression") / SIMPLE_WMIN_FIT

    # Ensure the destination directory exists
    destination_dir.mkdir(parents=True, exist_ok=True)

    # Copy the source directory to the destination
    dest_path = destination_dir / SIMPLE_WMIN_FIT
    shutil.copytree(source_dir, dest_path)

    # Get the fit path using the function to be tested
    fit_path = get_fit_path(SIMPLE_WMIN_FIT)

    # Check if the path exists and is of the correct type
    assert fit_path.exists()
    assert isinstance(fit_path, pathlib.Path)

    # Clean up the copied directory
    shutil.rmtree(dest_path)


def test_get_pdf_model_success():
    """Test successful retrieval of the PDF model when the file exists."""

    # Sample data to be returned by dill.load
    mock_pdf_model = "sample_pdf_model"

    # Mock the file path returned by get_fit_path
    mock_fit_path = pathlib.Path("/mock/path/to/fit")

    with patch("os.path.exists", return_value=True), patch(
        "builtins.open", mock_open(read_data="mock_data")
    ), patch("dill.load", return_value=mock_pdf_model), patch(
        "colibri.utils.get_fit_path", return_value=mock_fit_path
    ):

        result = get_pdf_model("mock_fit_name")

        assert result == mock_pdf_model


def test_get_pdf_model_file_not_found():
    """Test that FileNotFoundError is raised when the pdf_model.pkl does not exist."""

    # Mock the file path returned by get_fit_path
    mock_fit_path = pathlib.Path("/mock/path/to/fit")

    with patch("os.path.exists", return_value=False), patch(
        "colibri.utils.get_fit_path", return_value=mock_fit_path
    ):

        try:
            get_pdf_model("mock_fit_name")

        except FileNotFoundError as e:
            assert str(e) == "Could not find the pdf model for the fit mock_fit_name"


def test_get_full_posterior():
    """
    Test that get_full_posterior works correctly.
    """
    conda_prefix = os.getenv("CONDA_PREFIX")

    destination_dir = pathlib.Path(conda_prefix) / "share" / "colibri" / "results"
    source_dir = pathlib.Path("colibri/tests/regression") / SIMPLE_WMIN_FIT

    # Ensure the destination directory exists
    destination_dir.mkdir(parents=True, exist_ok=True)

    # Copy the source directory to the destination
    dest_path = destination_dir / SIMPLE_WMIN_FIT
    shutil.copytree(source_dir, dest_path)

    df = get_full_posterior(SIMPLE_WMIN_FIT)

    assert df is not None
    assert isinstance(df, pandas.core.frame.DataFrame)

    # Clean up the copied directory
    shutil.rmtree(dest_path)


def mock_bayesian_prior(array):
    # Mocked version of bayesian_prior
    return array


@pytest.mark.parametrize("dataset_input", [TEST_DATASET_HAD, TEST_DATASET])
def test_mask_fktable_array(dataset_input):
    """
    Test that masking of fktable arrays works as expected.
    """

    dataset = cAPI.dataset(**dataset_input)
    fktable = load_fktable(dataset.fkspecs[0]).with_cuts(dataset.cuts)

    # keep only flavours 1 and 2
    flavour_indices = [1, 2]

    # Without masking
    assert_allclose(mask_fktable_array(fktable), jnp.array(fktable.get_np_fktable()))

    # With masking
    masked_fktable = mask_fktable_array(fktable, flavour_indices)

    if fktable.hadronic:
        np_fktable = fktable.get_np_fktable()
        luminosity_mapping = fktable.luminosity_mapping

        mask_even = jnp.isin(luminosity_mapping[0::2], jnp.array(flavour_indices))
        mask_odd = jnp.isin(luminosity_mapping[1::2], jnp.array(flavour_indices))
        fk_arr_mask = mask_even * mask_odd

        expected_fktable = np_fktable[:, fk_arr_mask, :, :]

    else:
        np_fktable = fktable.get_np_fktable()
        lumi_indices = fktable.luminosity_mapping
        fk_arr_mask = jnp.isin(lumi_indices, jnp.array(flavour_indices))
        expected_fktable = np_fktable[:, fk_arr_mask, :]

    assert_allclose(masked_fktable, jnp.array(expected_fktable))


@pytest.mark.parametrize("dataset_input", [TEST_DATASET_HAD, TEST_DATASET])
def test_mask_luminosity_mapping(dataset_input):
    """
    Test that masking of luminosity mapping works as expected.
    """
    dataset = cAPI.dataset(**dataset_input)
    fktable = load_fktable(dataset.fkspecs[0]).with_cuts(dataset.cuts)

    # keep only flavours 1 and 2
    flavour_indices = [1, 2]

    # Without masking
    assert_allclose(
        mask_luminosity_mapping(fktable), jnp.array(fktable.luminosity_mapping)
    )

    # With masking
    masked_luminosity_mapping = mask_luminosity_mapping(fktable, flavour_indices)

    if fktable.hadronic:
        lumi_indices = fktable.luminosity_mapping
        mask_even = jnp.isin(lumi_indices[0::2], jnp.array(flavour_indices))
        mask_odd = jnp.isin(lumi_indices[1::2], jnp.array(flavour_indices))

        # for hadronic predictions pdfs enter in pair, hence product of two
        # boolean arrays and repeat by 2
        mask = jnp.repeat(mask_even * mask_odd, repeats=2)
        lumi_indices = lumi_indices[mask]

    else:
        lumi_indices = fktable.luminosity_mapping
        mask = jnp.isin(lumi_indices, jnp.array(flavour_indices))
        lumi_indices = lumi_indices[mask]

    assert_allclose(masked_luminosity_mapping, lumi_indices)


def test_likelihood_float_type(
    tmp_path,
):

    _pred_data = lambda x: jnp.ones(
        len(MOCK_CENTRAL_INV_COVMAT_INDEX.central_values)
    )  # Mock _pred_data
    FIT_XGRID = jnp.linspace(0, 1, 10)  # Mock FIT_XGRID
    output_path = tmp_path

    fast_kernel_arrays = jax.random.uniform(
        jax.random.PRNGKey(0), (10,)
    )  # Mock fast_kernel_arrays

    # Call the function under test
    likelihood_float_type(
        _pred_data=_pred_data,
        pdf_model=MOCK_PDF_MODEL,
        FIT_XGRID=FIT_XGRID,
        bayesian_prior=mock_bayesian_prior,
        output_path=output_path,
        central_inv_covmat_index=MOCK_CENTRAL_INV_COVMAT_INDEX,
        fast_kernel_arrays=fast_kernel_arrays,
    )

    # Assert that the dtype.txt file was created with correct dtype
    assert os.path.exists(tmp_path / "dtype.txt")


def test_identity_matrix():
    C = np.identity(3)
    expected = np.array([1.0, 1.0, 1.0, 1.0])
    result = compute_determinants_of_principal_minors(C)
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"


def test_single_element_matrix():
    C = np.array([[2]])
    expected = np.array([1.0, 2.0])
    result = compute_determinants_of_principal_minors(C)
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"


def test_non_psd_matrix():
    C = np.array([[2, -3], [-3, 1]])
    try:
        compute_determinants_of_principal_minors(C)
    except ValueError as e:
        assert str(e) == "Matrix is not positive semi-definite or symmetric."


def test_known_psd_matrix():
    C = np.array([[4, 2], [2, 3]])
    expected = np.array([1.0, 4.0, 8.0])
    result = compute_determinants_of_principal_minors(C)
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"


def test_large_psd_matrix():
    C = np.array([[4, 2, 0], [2, 3, 1], [0, 1, 2]])
    expected = np.array([1.0, 4.0, 8.0, 12.0])
    result = compute_determinants_of_principal_minors(C)
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"


def test_single_value():
    a = np.array([1.0, 2.0, 3.0])
    v = np.array([1.1])
    result = closest_indices(a, v, atol=0.2)
    expected = np.array([0])
    np.testing.assert_array_equal(result, expected)


def test_multiple_values():
    a = np.array([1.0, 2.0, 3.0])
    v = np.array([1.1, 3.0])
    result = closest_indices(a, v, atol=0.2)
    expected = np.array([0, 2])
    np.testing.assert_array_equal(result, expected)


def test_no_close_values():
    a = np.array([1.0, 2.0, 3.0])
    v = np.array([4.0])
    result = closest_indices(a, v, atol=0.2)
    expected = np.array([])  # No close values
    np.testing.assert_array_equal(result, expected)


def test_exact_match():
    a = np.array([1.0, 2.0, 3.0])
    v = np.array([1.0, 2.0, 3.0])
    result = closest_indices(a, v, atol=1e-7)
    expected = np.array([0, 1, 2])
    np.testing.assert_array_equal(result, expected)


def test_atol_effect():
    a = np.array([1.0, 2.0, 3.0])
    v = np.array([2.1])
    result = closest_indices(a, v, atol=0.09)  # Should not match 2.0 due to tight atol
    expected = np.array([])  # No match because atol is small
    np.testing.assert_array_equal(result, expected)

    result = closest_indices(a, v, atol=0.11)  # Now 2.1 is close enough to 2.0
    expected = np.array([1])
    np.testing.assert_array_equal(result, expected)


def test_identity_matrix():
    C = np.identity(3)
    expected = np.array([1.0, 1.0, 1.0, 1.0])
    result = compute_determinants_of_principal_minors(C)
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"


def test_single_element_matrix():
    C = np.array([[2]])
    expected = np.array([1.0, 2.0])
    result = compute_determinants_of_principal_minors(C)
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"


def test_non_psd_matrix():
    C = np.array([[2, -3], [-3, 1]])
    try:
        compute_determinants_of_principal_minors(C)
    except ValueError as e:
        assert str(e) == "Matrix is not positive semi-definite or symmetric."


def test_known_psd_matrix():
    C = np.array([[4, 2], [2, 3]])
    expected = np.array([1.0, 4.0, 8.0])
    result = compute_determinants_of_principal_minors(C)
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"


def test_large_psd_matrix():
    C = np.array([[4, 2, 0], [2, 3, 1], [0, 1, 2]])
    expected = np.array([1.0, 4.0, 8.0, 12.0])
    result = compute_determinants_of_principal_minors(C)
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"
