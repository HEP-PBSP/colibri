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
from colibri.api import API as cAPI
from colibri.tests.conftest import (
    MOCK_CENTRAL_INV_COVMAT_INDEX,
    MOCK_PDF_MODEL,
    TEST_DATASET_HAD,
    TEST_DATASET,
)
from colibri.utils import (
    cast_to_numpy,
    get_fit_path,
    get_full_posterior,
    get_pdf_model,
    likelihood_float_type,
    mask_fktable_array,
    mask_luminosity_mapping,
    ultranest_ns_fit_resampler,
    write_resampled_bayesian_fit,
    compute_determinants_of_principal_minors,
)
from colibri.constants import LHAPDF_XGRID, EXPORT_LABELS
from validphys.fkparser import load_fktable


SIMPLE_WMIN_FIT = "wmin_bayes_dis"


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

    assert "please run the bayesian fit first" in str(exc_info.value)
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
    n_replicas = 2
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
            n_replicas=n_replicas,
            resampled_fit_name=resampled_fit_name,
            parametrisation_scale=parametrisation_scale,
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


# @mock.patch("builtins.open", mock.mock_open(read_data=b"binary data"), new_callable=mock.mock_open)
@mock.patch("builtins.open", new_callable=mock.mock_open, read_data=b"binary data")
@mock.patch("dill.load")
@mock.patch("colibri.utils.os.system")
@mock.patch("colibri.utils.os.path.exists")
@mock.patch("colibri.utils.os.mkdir")
@mock.patch("colibri.utils.write_exportgrid")
def test_write_resampled_bayesian_fit_with_replica_range(
    mock_write_exportgrid,
    mock_mkdir,
    mock_exists,
    mock_os_system,
    mock_dill_load,
    mock_open,
):
    # Setup mock parameters
    fit_path = Path("/fake/fit/path")
    resampled_fit_path = Path("/fake/resampled/path")
    resampled_posterior = np.array([[0.1, 0.2], [0.3, 0.4]])
    n_replicas = 2
    resampled_fit_name = "test_grid"
    parametrisation_scale = 1.0
    replica_range = [0]  # Test with a specific range

    # Mock pdf_model and parameter names
    mock_pdf_model = MagicMock()
    mock_pdf_model.param_names = ["param1", "param2"]
    mock_pdf_model.grid_values_func.return_value = lambda params: [
        params[0] + 1,
        params[1] + 1,
    ]
    mock_dill_load.return_value = mock_pdf_model

    # Ensure os.path.exists returns True for necessary paths
    mock_exists.return_value = False

    # Mock Path().is_dir() to return True for the resampled path
    with patch.object(Path, "is_dir", return_value=True):

        # Run the function with the replica_range parameter
        write_resampled_bayesian_fit(
            resampled_posterior=resampled_posterior,
            fit_path=fit_path,
            resampled_fit_path=resampled_fit_path,
            n_replicas=n_replicas,
            resampled_fit_name=resampled_fit_name,
            parametrisation_scale=parametrisation_scale,
            replica_range=replica_range,  # Providing replica_range
        )

    # Ensure the correct range of replicas is processed
    assert mock_write_exportgrid.call_count == len(
        replica_range
    ), f"Expected {len(replica_range)} calls to write_exportgrid, but got {mock_write_exportgrid.call_count}"

    # Check that os.mkdir was called twice (once for new_rep_path and once for replica_index_path)
    # `new_rep_path` is resampled_fit_path / "replicas"
    new_rep_path = resampled_fit_path / "replicas"
    # `replica_index_path` is new_rep_path / f"replica_{i+1}" for each replica (here we test for i=0)
    replica_index_path = new_rep_path / "replica_1"

    # Assert that os.mkdir was called for both directories
    mock_mkdir.assert_any_call(new_rep_path)
    mock_mkdir.assert_any_call(replica_index_path)


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
