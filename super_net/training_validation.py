"""
TODO
"""
import jax
import jax.numpy as jnp

from reportengine import collect
from reportengine.configparser import ConfigError

from dataclasses import dataclass
from super_net.commondata_utils import CentralCovmatIndex


@dataclass(frozen=True)
class TrainCentralCovmatIndex(CentralCovmatIndex):
    n_training_points: int


@dataclass(frozen=True)
class ValidationCentralCovmatIndex(CentralCovmatIndex):
    n_validation_points: int


@dataclass(frozen=True)
class TrainValidationSplit:
    training_data: TrainCentralCovmatIndex = None
    validation_data: ValidationCentralCovmatIndex = None


def make_data_values(
    central_covmat_index: jnp.array,
    trval_seed: jnp.array,
    hyperopt: bool = False,
    bayesian_fit: bool = False,
    test_size: float = 0.2,
):
    """
    TODO
    """

    if bayesian_fit:
        fit_data = TrainCentralCovmatIndex(
            central_covmat_index.to_dict(),
            n_training_points=len(central_covmat_index.central_values),
        )
        return TrainValidationSplit(training_data=fit_data)

    if hyperopt:
        raise ConfigError("hyperopt not implemented yet")

    central_values = central_covmat_index.central_values
    covmat = central_covmat_index.covmat
    central_values_indices = central_covmat_index.central_values_idx

    # shuffle indices
    permuted_indices = jax.random.permutation(trval_seed, central_values_indices)

    # determine split point
    split_point = int(central_values.shape[0] * (1 - test_size))

    # split indices
    indices_train = permuted_indices[:split_point]
    indices_validation = permuted_indices[split_point:]

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

    return TrainValidationSplit(
        training_data=training_data, validation_data=validation_data
    )


"""
Collect over trval and replica indices.
"""
mc_replicas_make_data_values = collect(
    "make_data_values", ("trval_replica_indices",)
)
