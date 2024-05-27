"""
Module for testing the utils module.
"""

import os
import pathlib
import shutil
from unittest.mock import MagicMock, mock_open, patch

import jax
import numpy as np
import pandas
import pytest
from colibri.utils import (
    cast_to_numpy,
    get_fit_path,
    get_flow,
    get_full_posterior,
    get_pdf_model,
)

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
    if not conda_prefix:
        raise EnvironmentError("CONDA_PREFIX environment variable is not set")

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


def test_get_pdf_model():
    """
    Tests that get_pdf_model works correctly.
    """
    conda_prefix = os.getenv("CONDA_PREFIX")
    if not conda_prefix:
        raise EnvironmentError("CONDA_PREFIX environment variable is not set")

    destination_dir = pathlib.Path(conda_prefix) / "share" / "colibri" / "results"
    source_dir = pathlib.Path("colibri/tests/regression") / SIMPLE_WMIN_FIT

    # Ensure the destination directory exists
    destination_dir.mkdir(parents=True, exist_ok=True)

    # Copy the source directory to the destination
    dest_path = destination_dir / SIMPLE_WMIN_FIT
    shutil.copytree(source_dir, dest_path)

    # Check that loading pdf model from .pkl file works
    pdf_model = get_pdf_model(SIMPLE_WMIN_FIT)

    # check trivial stuff assotiated with the pdf model stored in SINGLE_WMIN_FIT
    assert pdf_model is not None
    assert pdf_model.n_basis == 10
    assert pdf_model.param_names == [
        "w_1",
        "w_2",
        "w_3",
        "w_4",
        "w_5",
        "w_6",
        "w_7",
        "w_8",
        "w_9",
        "w_10",
    ]
    assert pdf_model.name == "weight mininisation PDF model"

    # Clean up the copied directory
    shutil.rmtree(dest_path)


def test_get_full_posterior():
    """
    Test that get_full_posterior works correctly.
    """
    conda_prefix = os.getenv("CONDA_PREFIX")
    if not conda_prefix:
        raise EnvironmentError("CONDA_PREFIX environment variable is not set")

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


@patch("colibri.utils.os.path.exists")
@patch("colibri.utils.open", new_callable=mock_open)
@patch("colibri.utils.dill.load")
@patch("colibri.utils.get_fit_path")
def test_get_flow(mock_get_fit_path, mock_dill_load, mock_open, mock_exists):
    """
    Test that get_flow works correctly.
    """
    # Arrange
    colibri_fit = "test_fit"
    fit_path = MagicMock()
    norm_flow_path = fit_path / "norm_flow.pkl"

    mock_get_fit_path.return_value = fit_path
    mock_exists.return_value = True
    expected_flow = MagicMock()
    mock_dill_load.return_value = expected_flow

    # Act
    result = get_flow(colibri_fit)

    # Assert
    mock_get_fit_path.assert_called_once_with(colibri_fit)
    mock_exists.assert_called_once_with(norm_flow_path)
    mock_open.assert_called_once_with(norm_flow_path, "rb")
    mock_dill_load.assert_called_once()
    assert result == expected_flow


@patch("colibri.utils.os.path.exists")
@patch("colibri.utils.get_fit_path")
def test_get_flow_file_not_found(mock_get_fit_path, mock_exists):
    """
    Test that get_flow raises FileNotFoundError when the file is not found.
    """
    colibri_fit = "test_fit"
    fit_path = MagicMock()
    norm_flow_path = fit_path / "norm_flow.pkl"

    mock_get_fit_path.return_value = fit_path
    mock_exists.return_value = False

    # Act and Assert
    with pytest.raises(FileNotFoundError) as excinfo:
        get_flow(colibri_fit)

    assert str(excinfo.typename) == "FileNotFoundError"
    mock_get_fit_path.assert_called_once_with(colibri_fit)
    mock_exists.assert_called_once_with(norm_flow_path)
