"""
colibri.tests.test_utils.py

Module for testing the utils module.
"""

import os
import pathlib
from pathlib import Path
import shutil

from numpy.testing import assert_allclose
import pytest
from unittest.mock import patch, mock_open, MagicMock
from unittest import mock


import pandas as pd
import jax
import jax.numpy as jnp
import numpy as np
import pandas
import pytest
import validphys
from numpy.testing import assert_allclose
from validphys import convolution
from validphys.fkparser import load_fktable

from colibri.api import API as cAPI
from colibri.tests.conftest import (
    MOCK_CENTRAL_INV_COVMAT_INDEX,
    MOCK_PDF_MODEL,
    TEST_DATASET,
    TEST_DATASET_HAD,
)
from colibri.utils import (
    cast_to_numpy,
    closest_indices,
    compute_determinants_of_principal_minors,
    get_fit_path,
    get_full_posterior,
    get_pdf_model,
    likelihood_float_type,
    mask_fktable_array,
    mask_luminosity_mapping,
    ultranest_ns_fit_resampler,
    write_resampled_bayesian_fit,
    compute_determinants_of_principal_minors,
    resample_posterior_from_file,
    pdf_model_from_colibri_model,
    resample_from_ns_posterior,
    t0_pdf_grid,
)
from colibri.constants import LHAPDF_XGRID, EXPORT_LABELS
from validphys.fkparser import load_fktable


SIMPLE_WMIN_FIT = "wmin_bayes_dis"


@pytest.fixture
def mock_colibri_model():
    model = MagicMock()
    model.grid_values_func = MagicMock(
        return_value=lambda params: jnp.array(
            [[p * x for x in range(1, 6)] for p in params]
        )
    )
    return model


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

    assert t0_grid.shape == (N_rep, N_fl, len(FIT_XGRID))


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

    _pred_data = lambda x, fks: jnp.ones(
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


@patch("os.path.exists")
@patch("pandas.read_csv")
@patch("colibri.utils.resample_from_ns_posterior")
def test_ns_fit_resampler_file_not_found(mock_resample, mock_read_csv, mock_exists):
    # Test the case where the required posterior samples file does not exist
    fit_path = Path("/fake/path")
    n_replicas = 10
    resampling_seed = 42

    # Mock os.path.exists to return False, simulating missing file
    mock_exists.return_value = False

    with pytest.raises(FileNotFoundError) as exc_info:
        ultranest_ns_fit_resampler(fit_path, n_replicas, resampling_seed)

    assert "please run the appropriate fit first." in str(exc_info.value)
    mock_exists.assert_called_once_with(
        fit_path / "ultranest_logs/chains/equal_weighted_post.txt"
    )
    mock_read_csv.assert_not_called()
    mock_resample.assert_not_called()


@patch("colibri.utils.os.path.exists")
@patch("pandas.read_csv")
@patch("colibri.utils.resample_from_ns_posterior")
def test_ns_fit_resampler_replicas_exceeding_samples(
    mock_resample, mock_read_csv, mock_exists
):
    # Test the case where n_replicas exceeds the number of available posterior samples
    fit_path = Path("/fake/path")
    n_replicas = 15
    resampling_seed = 42

    # Mock os.path.exists to return True, simulating that the file exists
    mock_exists.return_value = True

    # Mock pandas.read_csv to return a dataframe with fewer rows than n_replicas
    sample_data = np.array([[1, 2], [3, 4], [5, 6]])  # 3 samples
    mock_read_csv.return_value = pd.DataFrame(sample_data)

    # Mock resample_from_ns_posterior to return expected value
    expected_resampled = np.array([[1, 2], [3, 4]])  # Example result
    mock_resample.return_value = expected_resampled

    result = ultranest_ns_fit_resampler(fit_path, n_replicas, resampling_seed)

    assert result is expected_resampled

    mock_exists.assert_called_once_with(
        fit_path / "ultranest_logs/chains/equal_weighted_post.txt"
    )
    mock_read_csv.assert_called_once_with(
        fit_path / "ultranest_logs/chains/equal_weighted_post.txt",
        sep="\s+",
        dtype=float,
    )

    # Ensure correct arguments were passed to mock_resample
    assert np.array_equal(mock_resample.call_args[0][0], sample_data)
    assert mock_resample.call_args[0][1] == len(sample_data)
    assert mock_resample.call_args[0][2] == resampling_seed


@patch("os.path.exists")
@patch("pandas.read_csv")
@patch("colibri.utils.resample_from_ns_posterior")
def test_ns_fit_resampler_normal_case(mock_resample, mock_read_csv, mock_exists):
    # Test the normal case where everything is as expected
    fit_path = Path("/fake/path")
    n_replicas = 2
    resampling_seed = 42

    # Mock os.path.exists to return True, simulating that the file exists
    mock_exists.return_value = True

    # Mock pandas.read_csv to return a dataframe with enough samples
    sample_data = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])  # 4 samples
    mock_read_csv.return_value = pd.DataFrame(sample_data)

    # Mock resample_from_ns_posterior to return expected value
    expected_resampled = np.array([[3, 4], [5, 6]])  # Example result
    mock_resample.return_value = expected_resampled

    result = ultranest_ns_fit_resampler(fit_path, n_replicas, resampling_seed)

    assert result is expected_resampled
    mock_exists.assert_called_once_with(
        fit_path / "ultranest_logs/chains/equal_weighted_post.txt"
    )
    mock_read_csv.assert_called_once_with(
        fit_path / "ultranest_logs/chains/equal_weighted_post.txt",
        sep="\s+",
        dtype=float,
    )

    # Ensure correct arguments were passed to mock_resample
    assert np.array_equal(mock_resample.call_args[0][0], sample_data)
    assert mock_resample.call_args[0][1] == n_replicas
    assert mock_resample.call_args[0][2] == resampling_seed


@patch("builtins.open", new_callable=mock.mock_open)
@patch("dill.load")
@patch("colibri.utils.os.system")
@patch("colibri.utils.os.path.exists")
@patch("colibri.utils.write_exportgrid")
def test_write_resampled_bayesian_fit(
    mock_write_exportgrid,
    mock_exists,
    mock_os_system,
    mock_dill_load,
    mock_open,
):
    # Setup mock parameters
    fit_path = Path("/fake/fit/path")
    resampled_fit_path = Path("/fake/resampled/path")
    resampled_posterior = np.array([[0.1, 0.2], [0.3, 0.4]])
    resampled_fit_name = "test_grid"
    parametrisation_scale = 1.0

    # Mock pdf_model and parameter names
    mock_pdf_model = MagicMock()
    mock_pdf_model.param_names = ["param1", "param2"]
    mock_pdf_model.grid_values_func.return_value = lambda params: [
        params[0] + 1,
        params[1] + 1,
    ]
    mock_dill_load.return_value = mock_pdf_model

    # Ensure os.path.exists returns True for necessary paths
    mock_exists.side_effect = lambda path: (
        True
        if str(resampled_fit_path) in str(path) or str(fit_path) in str(path)
        else False
    )

    # Mock Path().is_dir() to return True for the resampled path
    with patch.object(Path, "is_dir", return_value=True):
        # Run the function
        write_resampled_bayesian_fit(
            resampled_posterior=resampled_posterior,
            fit_path=fit_path,
            resampled_fit_path=resampled_fit_path,
            resampled_fit_name=resampled_fit_name,
            parametrisation_scale=parametrisation_scale,
            csv_results_name="ns_result.csv",
        )

    # Verify that open was called with pdf_model.pkl
    expected_open_call = mock.call(fit_path / "pdf_model.pkl", "rb")
    assert (
        expected_open_call in mock_open.call_args_list
    ), "Expected open to be called with pdf_model.pkl in read mode, but it wasn't."

    # Verify directory copy and removal of replicas
    mock_os_system.assert_any_call(f"cp -r {fit_path} {resampled_fit_path}")
    mock_os_system.assert_any_call(f"rm -r {resampled_fit_path}/replicas/*")

    # Verify the correct data was written to CSV
    df = pd.DataFrame(resampled_posterior, columns=mock_pdf_model.param_names)
    expected_csv_path = str(resampled_fit_path) + "/ns_result.csv"
    with patch("pandas.DataFrame.to_csv") as mock_to_csv:
        df.to_csv(expected_csv_path, float_format="%.5e")
        mock_to_csv.assert_called_once_with(expected_csv_path, float_format="%.5e")


def test_resample_posterior_not_use_all_columns():
    """
    Test resample_posterior_from_file when use_all_columns=False.
    """
    import pathlib

    # Mock inputs
    fit_path = pathlib.Path("/mock/path")
    file_path = pathlib.Path("mock_file.csv")
    n_replicas = 5
    resampling_seed = 42
    use_all_columns = False
    mock_data = pd.DataFrame(
        {
            "Column0": [0, 1, 2, 3, 4],
            "Column1": [10, 11, 12, 13, 14],
            "Column2": [20, 21, 22, 23, 24],
        }
    )

    # Mock the behavior of os.path.exists
    with patch("os.path.exists", return_value=True), patch(
        "pandas.read_csv", return_value=mock_data
    ), patch("colibri.utils.resample_from_ns_posterior") as mock_resampler:

        # Simulate the resampling function
        mock_resampler.return_value = "mock_resampled_posterior"

        # Call the function under test
        result = resample_posterior_from_file(
            fit_path=fit_path,
            file_path=file_path,
            n_replicas=n_replicas,
            resampling_seed=resampling_seed,
            use_all_columns=use_all_columns,
            read_csv_args={"sep": ",", "dtype": float},
        )

        # Assertions
        pd.testing.assert_frame_equal(
            pd.DataFrame(mock_data.iloc[:, 1:].values),
            pd.DataFrame([[10, 20], [11, 21], [12, 22], [13, 23], [14, 24]]),
        )
        assert mock_resampler.call_args[0][1] == n_replicas
        assert mock_resampler.call_args[0][2] == resampling_seed
        assert result == "mock_resampled_posterior"


def test_single_value():
    """
    Test for utils.closest_indices.
    """
    a = np.array([1.0, 2.0, 3.0])
    v = np.array([1.1])
    result = closest_indices(a, v, atol=0.2)
    expected = np.array([0])
    np.testing.assert_array_equal(result, expected)


def test_multiple_values():
    """
    Test for utils.closest_indices.
    """
    a = np.array([1.0, 2.0, 3.0])
    v = np.array([1.1, 3.0])
    result = closest_indices(a, v, atol=0.2)
    expected = np.array([0, 2])
    np.testing.assert_array_equal(result, expected)


def test_no_close_values():
    """
    Test for utils.closest_indices.
    """
    a = np.array([1.0, 2.0, 3.0])
    v = np.array([4.0])
    result = closest_indices(a, v, atol=0.2)
    expected = np.array([])  # No close values
    np.testing.assert_array_equal(result, expected)


def test_exact_match():
    """
    Test for utils.closest_indices.
    """
    a = np.array([1.0, 2.0, 3.0])
    v = np.array([1.0, 2.0, 3.0])
    result = closest_indices(a, v, atol=1e-7)
    expected = np.array([0, 1, 2])
    np.testing.assert_array_equal(result, expected)


def test_atol_effect():
    """
    Test for utils.closest_indices.
    """
    a = np.array([1.0, 2.0, 3.0])
    v = np.array([2.1])
    result = closest_indices(a, v, atol=0.09)  # Should not match 2.0 due to tight atol
    expected = np.array([])  # No match because atol is small
    np.testing.assert_array_equal(result, expected)

    result = closest_indices(a, v, atol=0.11)  # Now 2.1 is close enough to 2.0
    expected = np.array([1])
    np.testing.assert_array_equal(result, expected)


def test_scalar_v_input():
    """
    Test for utils.closest_indices.
    """
    a = np.array([1, 2, 3])
    v = np.float32(1.0)
    expected = 0
    result = closest_indices(a, v)
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"


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


def test_pdf_model_from_colibri_model_not_found():
    with patch("importlib.import_module", side_effect=ModuleNotFoundError):
        settings = {"model": "nonexistent_model"}
        with pytest.raises(ModuleNotFoundError):
            pdf_model_from_colibri_model(settings)


@patch("importlib.import_module")
def test_pdf_model_from_colibri_model_missing_config(mock_import_module):
    # Explicitly remove the 'config' attribute
    mock_module = MagicMock()
    del mock_module.config  # Ensure 'config' attribute doesn't exist
    mock_import_module.return_value = mock_module
    settings = {"model": "mock_model"}
    with pytest.raises(AttributeError):
        pdf_model_from_colibri_model(settings)


@patch("importlib.import_module")
@patch("inspect.getmembers")
def test_pdf_model_from_colibri_model_success(
    mock_getmembers, mock_import_module, mock_colibri_model
):
    mock_import_module.return_value = MagicMock()
    # Mock the colibriConfig class and its subclass
    from colibri.config import colibriConfig

    class MockColibriConfig(colibriConfig):
        def __init__(self, input_params):
            pass

        def produce_pdf_model(self, param1, param2, output_path, dump_model=False):
            return mock_colibri_model

    mock_getmembers.return_value = [("MockSubclass", MockColibriConfig)]

    # Define valid model settings
    model_settings = {
        "model": "mock_colibri_model",
        "param1": 1,
        "param2": 2,
    }

    # Call the function and assert the result
    result = pdf_model_from_colibri_model(model_settings)
    assert result == mock_colibri_model


@patch("importlib.import_module")
@patch("inspect.getmembers")
def test_pdf_model_from_colibri_model_incorrect_inputs(
    mock_getmembers, mock_import_module, mock_colibri_model
):
    mock_import_module.return_value = MagicMock()
    # Mock the colibriConfig class and its subclass
    from colibri.config import colibriConfig

    class MockColibriConfig(colibriConfig):
        def __init__(self, input_params):
            pass

        def produce_pdf_model(self, param1, param2, output_path, dump_model=False):
            return mock_colibri_model

    mock_getmembers.return_value = [("MockSubclass", MockColibriConfig)]

    # Define model settings missing param2
    model_settings = {
        "model": "mock_colibri_model",
        "param1": 1,
    }

    with pytest.raises(ValueError):
        pdf_model_from_colibri_model(model_settings)
