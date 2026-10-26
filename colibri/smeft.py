"""
colibri.smeft.py

SMEFT corrections to the theory predictions.

The SMEFT forward map (see :class:`colibri.forward_map.SMEFTForwardMap`)
rescales the SM predictions computed from the FK tables by a multiplicative
K-factor that depends on the Wilson coefficients :math:`c_i`:

.. math::

    T_{\\rm SMEFT} = T_{\\rm SM} \\left(1 + \\sum_i c_i L_i
                     + \\sum_{i,j} c_i c_j Q_{ij}\\right)

where :math:`L_i` (linear) and :math:`Q_{ij}` (quadratic, optional) are the
SMEFT responses normalised to the SM prediction they were computed with.

Where the responses come from is delegated to a backend, selected in the
runcard with ``smeft_settings: {backend: <name>}`` among the classes in
:data:`SMEFT_BACKENDS`. Which datasets get a correction, and at which order,
is set per dataset with the ``smeft_order`` key in ``dataset_inputs``.

A backend is a class whose constructor takes the ``backend_options`` of the
runcard, with one method::

    dataset_responses(dataset, order, parameters) -> (linear, quadratic)

- ``dataset``: ``validphys.core.DataSetSpec``.
- ``order``: perturbative order of the K-factors, as given in the runcard.
- ``parameters``: list of ``{"name": str, "linear_combination": {operator: coeff}}``.
- ``linear``: array of shape ``(n_params, n_points)``; ``quadratic``: array of
  shape ``(n_params, n_params, n_points)``, or None. Both over *all* the points
  of the dataset (no cuts) and relative to the SM.

Everything downstream of the backend (cuts, ordering of the data points, the
forward map) is backend independent, so adding a new source of K-factors
(e.g. SMEFiT) only requires such a class and an entry in :data:`SMEFT_BACKENDS`.

Currently available backends:

- ``simunet``: reads the SIMU-factor files shipped with SIMUnet.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from validphys.core import DataSetInput

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Datasets with a SMEFT correction
# ---------------------------------------------------------------------------


class SMEFTDataSetInput(DataSetInput):
    """
    A ``dataset_inputs`` entry with the ``smeft_order`` key, i.e. a dataset
    receiving a SMEFT correction. ``smeft_order`` is the perturbative order of
    the K-factors to use, as named by the backend (e.g. ``EFT_LO`` or
    ``EFT_NLO`` for SIMUnet).

    The dataset itself is unchanged, so it compares equal to the plain
    validphys ``DataSetInput`` it is built from.
    """

    def __init__(self, *, smeft_order, **kwargs):
        super().__init__(**kwargs)
        self.smeft_order = smeft_order

    @classmethod
    def from_dataset_input(cls, dataset_input, smeft_order):
        return cls(
            name=dataset_input.name,
            cfac=dataset_input.cfac,
            frac=dataset_input.frac,
            weight=dataset_input.weight,
            custom_group=dataset_input.custom_group,
            variant=dataset_input.variant,
            smeft_order=smeft_order,
        )


def smeft_orders(dataset_inputs) -> dict:
    """``{dataset_name: smeft_order}`` for the datasets with a SMEFT correction."""
    return {
        dsinput.name: dsinput.smeft_order
        for dsinput in dataset_inputs
        if getattr(dsinput, "smeft_order", None) is not None
    }


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SMEFTCorrections:
    """
    SMEFT responses for the whole fitted dataset, in the same order as the
    predictions returned by ``_pred_data`` (cuts applied).

    Attributes
    ----------
    parameter_names: tuple[str]

    linear: np.ndarray, shape (n_params, n_data)

    quadratic: np.ndarray, shape (n_params, n_params, n_data), optional
    """

    parameter_names: tuple
    linear: np.ndarray
    quadratic: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


class SIMUnetBackend:
    """
    K-factors from the SIMU-factor files of SIMUnet
    (https://github.com/scarlehoff/SIMUnet).

    Install it with ``pip install colibri[smeft]``. The SIMU-factor files live
    in the ``simudata`` folder of the SIMUnet repository, which a regular
    install does not include: SIMUnet only finds it by itself when installed
    from a source checkout (``pip install -e .``). Otherwise, point to the
    ``simudata`` folder of a SIMUnet checkout with ``simudata_path``.

    Parameters
    ----------
    simudata_path: str or pathlib.Path, optional
        Path to a SIMUnet ``simudata`` folder (containing ``simu_factors``).
        Defaults to the one SIMUnet finds next to its source tree.
    """

    def __init__(self, simudata_path=None):
        try:
            from simunet.loader import SIMUnetLoader
            from simunet.smeft import parameter_responses
        except ImportError as e:
            raise ImportError(
                "The 'simunet' SMEFT backend requires SIMUnet. Install it with "
                "`pip install colibri[smeft]` (or from "
                "https://github.com/scarlehoff/SIMUnet)."
            ) from e

        self._loader = SIMUnetLoader()
        if simudata_path is not None:
            self._loader.simudata_path = Path(simudata_path).expanduser().resolve()

        simu_factors = Path(self._loader.simudata_path) / "simu_factors"
        if not simu_factors.is_dir():
            raise FileNotFoundError(
                f"SIMU factor folder not found at {simu_factors}. SIMUnet only finds "
                "its simudata folder when installed from a source checkout "
                "(`pip install -e .`). Otherwise clone "
                "https://github.com/scarlehoff/SIMUnet and set "
                "smeft_settings: {backend_options: {simudata_path: <clone>/simudata}}."
            )
        self._parameter_responses = parameter_responses

    def dataset_responses(self, dataset, order, parameters):
        path = self._loader.get_simu_parameters_name_dict(dataset.name, [order])[order]
        factors = self._loader.load_simu_factors(path)
        if order not in factors:
            raise KeyError(
                f"Order '{order}' not found in the SIMU factor file of "
                f"{dataset.name} ({path}). Available keys: {list(factors)}"
            )
        linear = self._parameter_responses(list(parameters), factors[order])
        return np.asarray(linear, dtype=float), None


SMEFT_BACKENDS = {"simunet": SIMUnetBackend}
"""Available SMEFT backends, ``{name: backend class}`` (see the module docstring)."""


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------


def smeft_corrections(data, dataset_inputs, smeft_settings) -> SMEFTCorrections:
    """
    Collects the SMEFT responses of every dataset in ``data`` from the
    backend selected in ``smeft_settings``, applying the cuts. Only the
    datasets with a ``smeft_order`` key in ``dataset_inputs`` are corrected;
    the others get zero response (SM only).

    Parameters
    ----------
    data: validphys.core.DataGroupSpec

    dataset_inputs: list[validphys.core.DataSetInput]

    smeft_settings: dict
        Parsed ``smeft_settings``, see ``colibriConfig.parse_smeft_settings``.

    Returns
    -------
    SMEFTCorrections
    """
    orders = smeft_orders(dataset_inputs)
    if not orders:
        raise ValueError(
            "forward_map_type is smeft, but no dataset has a SMEFT correction. "
            "Add e.g. `smeft_order: EFT_NLO` to the relevant dataset_inputs entries."
        )

    backend = SMEFT_BACKENDS[smeft_settings["backend"]](
        **smeft_settings["backend_options"]
    )
    parameters = smeft_settings["parameters"]
    n_params = len(parameters)

    linear_blocks, quadratic_blocks = [], []
    for ds in data.datasets:
        ndata = ds.commondata.ndata
        cuts = ds.cuts.load() if ds.cuts is not None else None
        idx = np.arange(ndata) if cuts is None else np.asarray(cuts, dtype=int)

        if ds.name not in orders:
            linear_blocks.append(np.zeros((n_params, len(idx))))
            quadratic_blocks.append(None)
            continue

        order = orders[ds.name]
        log.info(f"Loading SMEFT {order} responses for {ds.name}")
        linear, quadratic = backend.dataset_responses(ds, order, parameters)

        linear = np.asarray(linear, dtype=float)
        if linear.shape != (n_params, ndata):
            raise ValueError(
                f"Linear SMEFT responses for {ds.name} have shape {linear.shape}, "
                f"expected {(n_params, ndata)}"
            )
        linear_blocks.append(linear[:, idx])

        if quadratic is None:
            quadratic_blocks.append(None)
        else:
            quadratic = np.asarray(quadratic, dtype=float)
            if quadratic.shape != (n_params, n_params, ndata):
                raise ValueError(
                    f"Quadratic SMEFT responses for {ds.name} have shape "
                    f"{quadratic.shape}, expected {(n_params, n_params, ndata)}"
                )
            quadratic_blocks.append(quadratic[..., idx])

    linear = np.concatenate(linear_blocks, axis=-1)

    if all(q is None for q in quadratic_blocks):
        quadratic = None
    else:
        quadratic = np.concatenate(
            [
                np.zeros((n_params, n_params, lin.shape[-1])) if q is None else q
                for lin, q in zip(linear_blocks, quadratic_blocks)
            ],
            axis=-1,
        )

    return SMEFTCorrections(
        parameter_names=tuple(p["name"] for p in parameters),
        linear=linear,
        quadratic=quadratic,
    )
