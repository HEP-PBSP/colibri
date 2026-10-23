"""
colibri.tests.test_smeft

Tests for the SMEFT K-factor backends and the smeft_corrections provider.
"""

import sys
import types
from unittest.mock import Mock

import numpy as np
import pytest

from colibri.smeft import (
    SMEFT_BACKENDS,
    DatasetResponses,
    SIMUnetBackend,
    SMEFTBackend,
    SMEFTParameter,
    SMEFTSettings,
    register_smeft_backend,
    smeft_corrections,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_dataset(name, ndata, cuts=None):
    ds = Mock()
    ds.name = name
    ds.commondata.ndata = ndata
    if cuts is None:
        ds.cuts = None
    else:
        ds.cuts.load.return_value = np.asarray(cuts)
    return ds


def _mock_data(*datasets):
    data = Mock()
    data.datasets = list(datasets)
    return data


class FakeBackend(SMEFTBackend):
    """Linear response of parameter i on point k of a dataset is 10*i + k."""

    quadratic = False

    def dataset_responses(self, dataset, order, parameters):
        n = dataset.commondata.ndata
        linear = np.array(
            [[10.0 * i + k for k in range(n)] for i in range(len(parameters))]
        )
        quadratic = None
        if self.quadratic:
            quadratic = np.ones((len(parameters), len(parameters), n))
        return DatasetResponses(linear=linear, quadratic=quadratic)


class FakeQuadraticBackend(FakeBackend):
    quadratic = True


@pytest.fixture
def fake_backends(monkeypatch):
    monkeypatch.setitem(SMEFT_BACKENDS, "fake", FakeBackend)
    monkeypatch.setitem(SMEFT_BACKENDS, "fake_quad", FakeQuadraticBackend)


def _settings(backend="fake", datasets=None, parameters=None):
    return SMEFTSettings(
        backend=backend,
        parameters=parameters
        or (SMEFTParameter("OtG", scale=50.0), SMEFTParameter("Y")),
        datasets={"SMEFT_DS": "EFT_NLO"} if datasets is None else datasets,
    )


# ---------------------------------------------------------------------------
# Settings dataclasses
# ---------------------------------------------------------------------------


def test_smeft_parameter_default_linear_combination():
    """A parameter without a linear combination is the operator with its name."""
    assert SMEFTParameter("OtG").linear_combination == {"OtG": 1.0}
    assert SMEFTParameter("Y", {"Olq1": 2.0}).linear_combination == {"Olq1": 2.0}


def test_smeft_settings_names_and_scales():
    settings = _settings()
    assert settings.parameter_names == ("OtG", "Y")
    np.testing.assert_array_equal(settings.scales, [50.0, 1.0])


# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------


def test_simunet_backend_is_registered():
    assert SMEFT_BACKENDS["simunet"] is SIMUnetBackend


def test_register_smeft_backend(monkeypatch):
    monkeypatch.setattr("colibri.smeft.SMEFT_BACKENDS", dict(SMEFT_BACKENDS))
    from colibri import smeft

    @register_smeft_backend("my_backend")
    class MyBackend(FakeBackend):
        pass

    assert smeft.SMEFT_BACKENDS["my_backend"] is MyBackend


def test_register_smeft_backend_rejects_non_backends():
    with pytest.raises(TypeError):

        @register_smeft_backend("not_a_backend")
        class NotABackend:
            pass


# ---------------------------------------------------------------------------
# smeft_corrections
# ---------------------------------------------------------------------------


def test_smeft_corrections_applies_cuts_and_zero_fills(fake_backends):
    """
    Responses follow the order of data.datasets, cuts are applied and datasets
    not listed in the settings get zero response.
    """
    data = _mock_data(
        _mock_dataset("SM_DS", ndata=3),
        _mock_dataset("SMEFT_DS", ndata=4, cuts=[0, 2, 3]),
    )

    corr = smeft_corrections(data, _settings())

    assert corr.parameter_names == ("OtG", "Y")
    np.testing.assert_array_equal(corr.scales, [50.0, 1.0])
    expected = np.array(
        [
            [0, 0, 0, 0.0, 2.0, 3.0],
            [0, 0, 0, 10.0, 12.0, 13.0],
        ]
    )
    np.testing.assert_array_equal(corr.linear, expected)
    assert corr.quadratic is None


def test_smeft_corrections_quadratic(fake_backends):
    """Quadratic responses are cut and zero-filled for SM-only datasets."""
    data = _mock_data(
        _mock_dataset("SMEFT_DS", ndata=3, cuts=[1, 2]),
        _mock_dataset("SM_DS", ndata=2),
    )

    corr = smeft_corrections(data, _settings(backend="fake_quad"))

    assert corr.quadratic.shape == (2, 2, 4)
    np.testing.assert_array_equal(corr.quadratic[..., :2], 1.0)
    np.testing.assert_array_equal(corr.quadratic[..., 2:], 0.0)


def test_smeft_corrections_unknown_dataset(fake_backends):
    data = _mock_data(_mock_dataset("SM_DS", ndata=3))
    with pytest.raises(ValueError, match="not part of the fit"):
        smeft_corrections(data, _settings())


def test_smeft_corrections_wrong_shape(fake_backends, monkeypatch):
    def bad_responses(self, dataset, order, parameters):
        return DatasetResponses(linear=np.zeros((len(parameters), 1)))

    monkeypatch.setattr(FakeBackend, "dataset_responses", bad_responses)
    data = _mock_data(_mock_dataset("SMEFT_DS", ndata=3))

    with pytest.raises(ValueError, match="expected"):
        smeft_corrections(data, _settings())


# ---------------------------------------------------------------------------
# SIMUnet backend
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_simunet(monkeypatch, tmp_path):
    """
    Install a minimal stand-in for the simunet package. Its loader looks for
    the SIMU factors in ``tmp_path/default_simudata`` (as a source install would).
    """
    default_simudata = tmp_path / "default_simudata"
    (default_simudata / "simu_factors").mkdir(parents=True)
    factors = {
        "metadata": {},
        "SM_fixed": [1.0, 1.0],
        "EFT_LO": {"SM": [2.0, 4.0], "OtG": [1.0, 2.0], "Olq1": [4.0, 4.0]},
    }

    class FakeLoader:
        def __init__(self):
            self.simudata_path = default_simudata

        def get_simu_parameters_name_dict(self, setname, names):
            return {n: f"SIMU_{setname}.yaml" for n in names}

        @staticmethod
        def load_simu_factors(path):
            return factors

    def parameter_responses(simu_parameters, cfactors):
        sm = np.asarray(cfactors["SM"])
        out = []
        for p in simu_parameters:
            resp = np.zeros_like(sm)
            for op, coeff in p.get("linear_combination", {p["name"]: 1.0}).items():
                resp += np.asarray(cfactors.get(op, 0.0)) / sm * coeff
            out.append(resp / p.get("scale", 1.0))
        return out

    simunet = types.ModuleType("simunet")
    loader = types.ModuleType("simunet.loader")
    loader.SIMUnetLoader = FakeLoader
    smeft = types.ModuleType("simunet.smeft")
    smeft.parameter_responses = parameter_responses
    monkeypatch.setitem(sys.modules, "simunet", simunet)
    monkeypatch.setitem(sys.modules, "simunet.loader", loader)
    monkeypatch.setitem(sys.modules, "simunet.smeft", smeft)


def test_simunet_backend_responses(fake_simunet):
    """Responses are K-factor / SM, combined linearly, without the scale."""
    backend = SIMUnetBackend()
    parameters = (
        SMEFTParameter("OtG", scale=50.0),
        SMEFTParameter("Y", {"Olq1": 0.5, "OtG": 1.0}, scale=200.0),
    )

    responses = backend.dataset_responses(Mock(name="ds"), "EFT_LO", parameters)

    np.testing.assert_allclose(responses.linear, [[0.5, 0.5], [1.5, 1.0]])
    assert responses.quadratic is None


def test_simunet_backend_missing_order(fake_simunet):
    backend = SIMUnetBackend()
    with pytest.raises(KeyError, match="EFT_NLO"):
        backend.dataset_responses(Mock(), "EFT_NLO", (SMEFTParameter("OtG"),))


def test_simunet_backend_simudata_path(fake_simunet, tmp_path):
    """simudata_path overrides the folder SIMUnet would use by itself."""
    custom = tmp_path / "custom_simudata"
    (custom / "simu_factors").mkdir(parents=True)

    backend = SIMUnetBackend(simudata_path=str(custom))

    assert backend._loader.simudata_path == custom.resolve()


def test_simunet_backend_missing_simudata(fake_simunet, tmp_path):
    """A clear error when the SIMU factors cannot be found (plain install)."""
    with pytest.raises(FileNotFoundError, match="simudata_path"):
        SIMUnetBackend(simudata_path=tmp_path / "nowhere")


def test_smeft_corrections_passes_backend_options(monkeypatch):
    """backend_options are passed to the backend constructor."""
    received = {}

    class OptionsBackend(FakeBackend):
        def __init__(self, path=None):
            received["path"] = path

    monkeypatch.setitem(SMEFT_BACKENDS, "options", OptionsBackend)
    settings = SMEFTSettings(
        backend="options",
        parameters=(SMEFTParameter("OtG"),),
        datasets={"SMEFT_DS": "EFT_LO"},
        backend_options={"path": "somewhere"},
    )

    smeft_corrections(_mock_data(_mock_dataset("SMEFT_DS", ndata=2)), settings)

    assert received == {"path": "somewhere"}


def test_simunet_backend_missing_package(monkeypatch):
    monkeypatch.setitem(sys.modules, "simunet", None)
    monkeypatch.setitem(sys.modules, "simunet.loader", None)
    with pytest.raises(ImportError, match="requires SIMUnet"):
        SIMUnetBackend()


# ---------------------------------------------------------------------------
# Resolution through the reportengine API
# ---------------------------------------------------------------------------


def test_smeft_corrections_from_api(fake_backends):
    """smeft_corrections resolves as a reportengine resource from runcard input."""
    from colibri.api import API
    from colibri.tests.conftest import TEST_DATASETS

    name = TEST_DATASETS["dataset_inputs"][0]["dataset"]
    smeft_settings = {
        "backend": "fake",
        "order": "EFT_NLO",
        "parameters": [{"name": "OtG"}],
        "datasets": [{"dataset": name}],
    }

    corr = API.smeft_corrections(**TEST_DATASETS, smeft_settings=smeft_settings)
    fk_arrays = API.fast_kernel_arrays(**TEST_DATASETS)

    assert corr.parameter_names == ("OtG",)
    assert corr.linear.shape == (1, fk_arrays[0][0].shape[0])
