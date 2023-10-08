"""
TODO
"""
import jax
import jax.numpy as jnp

from dataclasses import dataclass, asdict


def replica_seed(replica_index):
    """
    Generate a random integer given a replica_index.
    Note that each replica index has a unique key.
    """
    key = jax.random.PRNGKey(replica_index)
    randint = jax.random.randint(key, shape=(1,), minval=0, maxval=1e10)
    return int(randint)

def trval_seed(trval_index):
    """
    Returns a PRNGKey key given `trval_index` seed.
    """
    key = jax.random.PRNGKey(trval_index)
    return key


@dataclass(frozen=True)
class TrainValidationSplit:
    training: jnp.array
    validation: jnp.array

    def to_dict(self):
        return asdict(self)


def training_validation_split(indices, test_size, random_seed, shuffle_indices=True):
    """
    Performs training validation split on an array.

    Parameters
    ----------
    indices: jaxlib.xla_extension.Array

    test_size: float

    random_seed: jaxlib.xla_extension.Array
        PRNGKey, obtained as jax.random.PRNGKey(random_number)

    shuffle_indices: bool

    Returns
    -------
    dataclass
    """

    if shuffle_indices:
        # shuffle indices
        permuted_indices = jax.random.permutation(random_seed, indices)
    else:
        permuted_indices = indices

    # determine split point
    split_point = int(indices.shape[0] * (1 - test_size))

    # split indices
    indices_train = permuted_indices[:split_point]
    indices_validation = permuted_indices[split_point:]

    return TrainValidationSplit(
        training=indices_train, 
        validation=indices_validation
    )
