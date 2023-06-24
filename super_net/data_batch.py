import numpy as np
import numpy.random as npr


class DataBatch:
    def __init__(self, Ndat, batch_size=128, seed=0):
        self.Ndat = Ndat
        self.batch_size = batch_size
        self.seed = seed

    def data_batch_stream_index(self):
        """yields array of indices of specified shape.
        The indices are chosen at random via shuffling.
        """
        Ndat = self.Ndat
        batch_size = self.batch_size
        seed = self.seed

        if batch_size > Ndat:
            raise ValueError(
                f"size of batch = {batch_size} should be smaller or equal to the number of data {Ndat}"
            )

        rng = npr.RandomState(seed)
        num_complete_batches, leftover = divmod(Ndat, batch_size)
        # discard leftover to avoid the a slow down due to having to recompile make_chi2 functionm
        num_batches = num_complete_batches  # + bool(leftover)

        while True:
            perm = rng.permutation(Ndat)

            for i in range(num_batches):
                batch_idx = perm[i * batch_size : (i + 1) * batch_size]
                yield batch_idx

    def num_batches(self):
        Ndat = self.Ndat
        batch_size = self.batch_size
        seed = self.seed

        if batch_size > Ndat:
            raise ValueError(
                f"size of batch = {batch_size} should be smaller or equal to the number of data {Ndat}"
            )

        num_complete_batches, leftover = divmod(Ndat, batch_size)
        # discard leftover to avoid the a slow down due to having to recompile make_chi2 functionm
        num_batches = num_complete_batches  # + bool(leftover)
        return num_batches


def data_batch_stream_index(nr_training_points, batch_size, batch_seed):
    """yields array of indices of specified shape.
    The indices are chosen at random via shuffling.

    Parameters
    ----------

    """

    if batch_size > nr_training_points:
        raise ValueError(
            f"size of batch = {batch_size} should be smaller or equal to the number of data {nr_training_points}"
        )

    rng = npr.RandomState(batch_seed)
    num_complete_batches, leftover = divmod(nr_training_points, batch_size)
    # discard leftover to avoid the a slow down due to having to recompile make_chi2 functionm
    num_batches = num_complete_batches  # + bool(leftover)

    while True:
        perm = rng.permutation(nr_training_points)

        for i in range(num_batches):
            batch_idx = perm[i * batch_size : (i + 1) * batch_size]
            yield batch_idx


def num_batches(nr_training_points, batch_size):
    """
    Parameters
    ----------

    Returns
    -------
    """
    # this test should probably be decorator. to add in checks.py module
    if batch_size > nr_training_points:
        raise ValueError(
            f"size of batch = {batch_size} should be smaller or equal to the number of data {nr_training_points}"
        )

    num_complete_batches, leftover = divmod(nr_training_points, batch_size)
    # discard leftover to avoid the a slow down due to having to recompile make_chi2 functionm
    num_batches = num_complete_batches  # + bool(leftover)
    return num_batches


def data_batch_info(data_batch_stream_index, num_batches, batch_size):
    """ """
    return {
        "data_batch_stream_index": data_batch_stream_index,
        "num_batches": num_batches,
        "batch_size": batch_size,
    }
