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

Where the responses come from is delegated to an :class:`SMEFTBackend`.
Backends are registered by name with :func:`register_smeft_backend` and
selected in the runcard with ``smeft_settings: {backend: <name>}``.
Everything downstream of the backend (cuts, ordering of the data points,
fit-parameter scales, the forward map itself) is backend independent, so
adding a new source of K-factors (e.g. SMEFiT) only requires implementing
:meth:`SMEFTBackend.dataset_responses`.

Currently available backends:

- ``simunet``: reads the SIMU-factor files shipped with SIMUnet.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import NamedTuple, Optional

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SMEFTParameter:
    """
    A SMEFT fit parameter.

    Attributes
    ----------
    name: str
        Name of the fit parameter, as it appears in ``forward_map.param_names``.

    linear_combination: dict
        Mapping ``{operator: coefficient}`` defining the parameter as a linear
        combination of the operators known to the backend. Defaults to
        ``{name: 1.0}``, i.e. the parameter is a single operator.

    scale: float
        The fitted parameter is ``scale * c``, with ``c`` the physical
        Wilson coefficient. Useful to bring all parameters to O(1).
    """

    name: str
    linear_combination: dict = field(default_factory=dict)
    scale: float = 1.0

    def __post_init__(self):
        if not self.linear_combination:
            object.__setattr__(self, "linear_combination", {self.name: 1.0})


@dataclass(frozen=True)
class SMEFTSettings:
    """
    Parsed ``smeft_settings`` runcard namespace.

    Attributes
    ----------
    backend: str
        Name of the registered :class:`SMEFTBackend` providing the K-factors.

    parameters: tuple[SMEFTParameter]
        SMEFT fit parameters, in the order they are appended to the PDF
        parameters.

    datasets: dict
        Mapping ``{dataset_name: order}`` of the datasets receiving a SMEFT
        correction, and the perturbative order of the K-factors to use
        (e.g. ``EFT_LO`` or ``EFT_NLO`` for SIMUnet). All other datasets are
        treated as SM only.
    """

    backend: str
    parameters: tuple
    datasets: dict

    @property
    def parameter_names(self) -> tuple:
        return tuple(p.name for p in self.parameters)

    @property
    def scales(self) -> np.ndarray:
        return np.array([p.scale for p in self.parameters], dtype=float)


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------


class DatasetResponses(NamedTuple):
    """
    SMEFT responses of a single dataset, as returned by a backend.

    Both arrays run over *all* the points of the dataset (no cuts applied)
    and are relative to the SM, for unit physical Wilson coefficients.

    linear: np.ndarray, shape (n_params, n_points)
    quadratic: np.ndarray, shape (n_params, n_params, n_points), optional
    """

    linear: np.ndarray
    quadratic: Optional[np.ndarray] = None


@dataclass(frozen=True)
class SMEFTCorrections:
    """
    SMEFT responses for the whole fitted dataset, in the same order as the
    predictions returned by ``_pred_data`` (cuts applied).

    Attributes
    ----------
    parameter_names: tuple[str]

    scales: np.ndarray, shape (n_params,)
        Fit parameters are ``scales * c`` (see :class:`SMEFTParameter`).

    linear: np.ndarray, shape (n_params, n_data)

    quadratic: np.ndarray, shape (n_params, n_params, n_data), optional
    """

    parameter_names: tuple
    scales: np.ndarray
    linear: np.ndarray
    quadratic: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


class SMEFTBackend(ABC):
    """
    Source of SMEFT K-factors.

    A backend only needs to know how to produce the responses of one dataset
    for a given list of fit parameters. Cuts, ordering of the data and scales
    are handled by colibri.
    """

    @abstractmethod
    def dataset_responses(
        self, dataset, order: str, parameters: tuple
    ) -> DatasetResponses:
        """
        Parameters
        ----------
        dataset: validphys.core.DataSetSpec

        order: str
            Perturbative order of the K-factors, as given in the runcard.

        parameters: tuple[SMEFTParameter]

        Returns
        -------
        DatasetResponses
            Responses for all the (uncut) points of the dataset.
        """
        raise NotImplementedError


SMEFT_BACKENDS: dict = {}
"""Registry of available SMEFT backends, ``{name: SMEFTBackend subclass}``."""


def register_smeft_backend(name: str):
    """Class decorator registering an :class:`SMEFTBackend` under ``name``."""

    def decorator(cls):
        if not issubclass(cls, SMEFTBackend):
            raise TypeError(f"{cls.__name__} must be a subclass of SMEFTBackend")
        SMEFT_BACKENDS[name] = cls
        return cls

    return decorator


@register_smeft_backend("simunet")
class SIMUnetBackend(SMEFTBackend):
    """
    K-factors from the SIMU-factor files of SIMUnet
    (https://github.com/scarlehoff/SIMUnet).

    SIMUnet locates its ``simudata`` folder relative to its source tree, so
    it must be installed from a source checkout (``pip install -e .``).
    """

    def __init__(self):
        try:
            from simunet.loader import SIMUnetLoader
            from simunet.smeft import parameter_responses
        except ImportError as e:
            raise ImportError(
                "The 'simunet' SMEFT backend requires SIMUnet. Install it from a "
                "source checkout of https://github.com/scarlehoff/SIMUnet with "
                "`pip install -e .`"
            ) from e

        self._loader = SIMUnetLoader()
        self._parameter_responses = parameter_responses

    def dataset_responses(self, dataset, order, parameters):
        path = self._loader.get_simu_parameters_name_dict(dataset.name, [order])[order]
        factors = self._loader.load_simu_factors(path)
        if order not in factors:
            raise KeyError(
                f"Order '{order}' not found in the SIMU factor file of "
                f"{dataset.name} ({path}). Available keys: {list(factors)}"
            )
        # Scales are applied by colibri, so pass physical coefficients only.
        specs = [
            {"name": p.name, "linear_combination": dict(p.linear_combination)}
            for p in parameters
        ]
        linear = self._parameter_responses(specs, factors[order])
        return DatasetResponses(linear=np.asarray(linear, dtype=float))


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------


def smeft_corrections(data, smeft_settings: SMEFTSettings) -> SMEFTCorrections:
    """
    Collects the SMEFT responses of every dataset in ``data`` from the
    backend selected in ``smeft_settings``, applying the cuts. Datasets not
    listed in ``smeft_settings.datasets`` get zero response (SM only).

    Parameters
    ----------
    data: validphys.core.DataGroupSpec

    smeft_settings: SMEFTSettings

    Returns
    -------
    SMEFTCorrections
    """
    dataset_names = [ds.name for ds in data.datasets]
    unknown = set(smeft_settings.datasets) - set(dataset_names)
    if unknown:
        raise ValueError(
            f"Datasets {sorted(unknown)} in smeft_settings are not part of the fit. "
            f"Fitted datasets: {dataset_names}"
        )

    backend = SMEFT_BACKENDS[smeft_settings.backend]()
    parameters = smeft_settings.parameters
    n_params = len(parameters)

    linear_blocks, quadratic_blocks = [], []
    for ds in data.datasets:
        ndata = ds.commondata.ndata
        cuts = ds.cuts.load() if ds.cuts is not None else None
        idx = np.arange(ndata) if cuts is None else np.asarray(cuts, dtype=int)

        if ds.name not in smeft_settings.datasets:
            linear_blocks.append(np.zeros((n_params, len(idx))))
            quadratic_blocks.append(None)
            continue

        order = smeft_settings.datasets[ds.name]
        log.info(f"Loading SMEFT {order} responses for {ds.name}")
        responses = backend.dataset_responses(ds, order, parameters)

        linear = np.asarray(responses.linear, dtype=float)
        if linear.shape != (n_params, ndata):
            raise ValueError(
                f"Linear SMEFT responses for {ds.name} have shape {linear.shape}, "
                f"expected {(n_params, ndata)}"
            )
        linear_blocks.append(linear[:, idx])

        if responses.quadratic is None:
            quadratic_blocks.append(None)
        else:
            quadratic = np.asarray(responses.quadratic, dtype=float)
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
        parameter_names=smeft_settings.parameter_names,
        scales=smeft_settings.scales,
        linear=linear,
        quadratic=quadratic,
    )
