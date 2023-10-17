"""
TODO
"""
import jax.numpy as jnp
from dataclasses import dataclass

from reportengine import collect
from reportengine.configparser import ConfigError

from super_net.commondata_utils import CentralCovmatIndex
from super_net.monte_carlo_utils import training_validation_split, TrainValidationSplit


@dataclass(frozen=True)
class TrainCentralCovmatIndex(CentralCovmatIndex):
    n_training_points: int


@dataclass(frozen=True)
class ValidationCentralCovmatIndex(CentralCovmatIndex):
    n_validation_points: int


@dataclass(frozen=True)
class MakeDataValues:
    training_data: TrainCentralCovmatIndex = None
    validation_data: ValidationCentralCovmatIndex = None


def make_data_values(
    central_covmat_index: dataclass,
    trval_seed: jnp.array,
    hyperopt: bool = False,
    bayesian_fit: bool = False,
    test_size: float = 0.2,
    shuffle_indices: bool = True,
):
    """
    Validphys provider for data values pre fit.
    Contains the logic for:

    - Monte Carlo fit
    - Monte Carlo hyperopt fit
    - Bayesian fit.

    Parameters
    ----------
    central_covmat_index: dataclass
    trval_seed: jnp.array,
    hyperopt: bool = False,
    bayesian_fit: bool = False,
    test_size: float = 0.2,
    shuffle_indices: bool = True

    Returns
    -------
    dataclass
    """

    if bayesian_fit:
        # no tr/val split needed for a bayesian fit
        fit_data = TrainCentralCovmatIndex(
            **central_covmat_index.to_dict(),
            n_training_points=len(central_covmat_index.central_values),
        )
        return MakeDataValues(training_data=fit_data)

    if hyperopt:
        raise ConfigError("hyperopt not implemented yet")

    central_values = central_covmat_index.central_values
    covmat = central_covmat_index.covmat
    central_values_indices = central_covmat_index.central_values_idx

    # perform tr/val split
    trval_split = training_validation_split(
        central_values_indices, test_size, trval_seed, shuffle_indices
    )
    indices_train = trval_split.training
    indices_validation = trval_split.validation

    # split data
    training_data = TrainCentralCovmatIndex(
        central_values=central_values[indices_train],
        covmat=covmat[indices_train][:, indices_train],
        central_values_idx=indices_train,
        n_training_points=len(indices_train),
    )

    validation_data = ValidationCentralCovmatIndex(
        central_values=central_values[indices_validation],
        covmat=covmat[indices_validation][:, indices_validation],
        central_values_idx=indices_validation,
        n_validation_points=len(indices_validation),
    )

    return MakeDataValues(training_data=training_data, validation_data=validation_data)


"""
Collect over trval and replica indices.
"""
mc_replicas_make_data_values = collect("make_data_values", ("trval_replica_indices",))


@dataclass(frozen=True)
class PosdataTrainValidationSplit(TrainValidationSplit):
    n_training: int
    n_validation: int


def make_posdata_split(
    posdatasets, trval_seed, test_size=0.2, shuffle_indices=True, bayesian_fit=False
):
    """
    TODO
    note: same seed as for data tr/val split.
    """

    ndata_pos = jnp.sum(
        jnp.array(
            [
                pos_ds.load_commondata().with_cuts(pos_ds.cuts).ndata
                for pos_ds in posdatasets
            ]
        )
    )
    indices = jnp.arange(ndata_pos)

    # no tr/vl split needed for bayesian fit
    if bayesian_fit:
        return PosdataTrainValidationSplit(
            training=indices,
            validation=None,
            n_training=len(indices),
            n_validation=None,
        )

    trval_split = training_validation_split(
        indices, test_size, trval_seed, shuffle_indices
    )
    n_training = len(trval_split.training)
    n_validation = len(trval_split.validation)

    return PosdataTrainValidationSplit(
        **trval_split.to_dict(), n_training=n_training, n_validation=n_validation
    )
