"""
super_net.utils.py

Module containing several utils for PDF fits.

Author: Mark N. Costantini
Date: 11.11.2023
"""

import jax
import jax.numpy as jnp

from dataclasses import dataclass, asdict

from super_net.constants import XGRID
from super_net.covmats import (
    dataset_inputs_t0_covmat_from_systematics,
    dataset_inputs_covmat_from_systematics,
)

from validphys import convolution
from validphys.pseudodata import make_replica, indexed_make_replica


FLAVOURS_ID_MAPPINGS = {
    0: "photon",
    1: "\Sigma",
    2: "g",
    3: "V",
    4: "V3",
    5: "V8",
    6: "V15",
    7: "V24",
    8: "V35",
    9: "T3",
    10: "T8",
    11: "T15",
    12: "T24",
    13: "T35",
}

FLAVOUR_TO_ID_MAPPING = {val: key for (key, val) in FLAVOURS_ID_MAPPINGS.items()}


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

    return TrainValidationSplit(training=indices_train, validation=indices_validation)


def t0_pdf_grid(t0pdfset, Q0=1.65):
    """
    Computes the t0 pdf grid in the evolution basis.

    Parameters
    ----------
    t0pdfset: validphys.core.PDF

    Q0: float, default is 1.65

    Returns
    -------
    t0grid: jnp.array
        t0 grid, is N_rep x N_fl x N_x
    """

    t0grid = jnp.array(
        convolution.evolution.grid_values(
            t0pdfset, convolution.FK_FLAVOURS, XGRID, [Q0]
        ).squeeze(-1)
    )
    return t0grid


def closure_test_pdf_grid(closure_test_pdf, Q0=1.65):
    """
    Computes the closure_test_pdf grid in the evolution basis.

    Parameters
    ----------
    closure_test_pdf: validphys.core.PDF

    Q0: float, default is 1.65

    Returns
    -------
    grid: jnp.array
        grid, is N_rep x N_fl x N_x
    """

    grid = jnp.array(
        convolution.evolution.grid_values(
            closure_test_pdf, convolution.FK_FLAVOURS, XGRID, [Q0]
        ).squeeze(-1)
    )
    return grid


def resample_from_ns_posterior(
    samples, n_posterior_samples=1000, posterior_resampling_seed=123456
):
    """
    TODO
    """

    current_samples = samples.copy()

    rng = jax.random.PRNGKey(posterior_resampling_seed)

    resampled_samples = jax.random.choice(
        rng, current_samples, (n_posterior_samples,), replace=False
    )

    return resampled_samples


def closure_test_central_pdf_grid(closure_test_pdf_grid):
    """
    Returns the central replica of the closure test pdf grid.
    """
    return closure_test_pdf_grid[0]


def make_level1_data(data, commondata_tuple, filterseed, data_index, fakedata, use_jax_random=False):
    """
    Given a tuple of commondata instances, return the
    same tuple with central values replaced with a sample of a multivariate
    gaussian centred at the central value and with covariance matrix
    constructed with the following procedure:

        - if fakedata=True, use dataset_inputs_t0_covmat_from_systematics
        - else use the experimental covariance matrix, i.e. dataset_inputs_covmat_from_systematics

    Note that the two covariance matrices above only differ in the MULT uncertainties.



    Parameters
    ----------

    data : validphys.core.DataGroupSpec

    commondata_tuple : tuple or list
                        list of validphys.coredata.CommonData instances corresponding to
                        all datasets within one experiment. The central value is replaced
                        by Level 0 fake data. Cuts already applied.

    filterseed : int
                random seed used for the generation of Level 1 data

    data_index : pandas.MultiIndex

    fakedata : bool, default is False

    Returns
    -------
    list
        list of validphys.coredata.CommonData instances corresponding to
        all datasets within one experiment. The central value is replaced
        by Level 1 fake data.

    Example
    -------

    >>> from validphys.api import API
    >>> dataset='NMC'
    >>> l1_cd = API.make_level1_data(dataset_inputs = [{"dataset":dataset}],use_cuts="internal", theoryid=200,
                             fakepdf = "NNPDF40_nnlo_as_01180",filterseed=1)
    >>> l1_cd
    [CommonData(setname='NMC', ndata=204, commondataproc='DIS_NCE', nkin=3, nsys=16)]
    """
    # if fakedata:
    #     covmat = dataset_inputs_t0_covmat_from_systematics(
    #         data, commondata_tuple, super_net_dataset_inputs_t0_predictions=None
    #     )
    # else:
    #     covmat = dataset_inputs_covmat_from_systematics(data, commondata_tuple)

    # TEST: use only additive uncertainties
    from validphys import covmats

    covmat = jnp.array(
    covmats.dataset_inputs_covmat_from_systematics(
        commondata_tuple,
        data.dsinputs,
        use_weights_in_covmat=False,
        norm_threshold=None,
        _list_of_central_values=None,
        _only_additive=True,
    )
)
    # ================== generation of Level1 data ======================#
    if use_jax_random:
        rng = jax.random.PRNGKey(filterseed)
        central_values = jnp.array([cd.central_values for cd in commondata_tuple]).flatten()
        level1_data = jax.random.multivariate_normal(rng, central_values, covmat)
    else:
        level1_data = make_replica(
            commondata_tuple, filterseed, covmat, sep_mult=False, genrep=True
        )

    indexed_level1_data = indexed_make_replica(data_index, level1_data)

    dataset_order = {cd.setname: i for i, cd in enumerate(commondata_tuple)}

    # ===== create commondata instances with central values given by pseudo_data =====#
    level1_commondata_dict = {c.setname: c for c in commondata_tuple}
    level1_commondata_instances_wc = []

    for xx, grp in indexed_level1_data.groupby("dataset"):
        level1_commondata_instances_wc.append(
            level1_commondata_dict[xx].with_central_value(grp.values)
        )
    # sort back so as to mantain same order as in commondata_tuple
    level1_commondata_instances_wc.sort(key=lambda x: dataset_order[x.setname])

    return level1_commondata_instances_wc
