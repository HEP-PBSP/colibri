"""
TODO
"""
import os
import pathlib
import logging

import lhapdf

import jax
import jax.numpy as jnp

from validphys.lhio import (
    load_all_replicas,
    write_replica,
    rep_matrix,
)

from wmin.checks import check_wminpdfset_is_montecarlo

log = logging.getLogger(__name__)


def wmin_grid_seed(wmin_grid_index):
    """
    Wmin PRNGKey used for the random choice of wmin replicas to be used
    in the wmin parametrisation and random choice of the central wmin
    replica
    """
    key = jax.random.PRNGKey(wmin_grid_index)
    return key


def weights_initializer_provider(
    weights_initializer="zeros",
    weights_seed=0xABCDEF,
    uniform_minval=-0.1,
    uniform_maxval=0.1,
):
    """
    Function responsible for the initialization of the weights in a weight minimization fit.

    Parameters
    ----------
    weights_initializer: str, default is 'zeros'
            the available options are: ('zeros', 'normal', 'uniform')
            if an unknown option is specified, the 'zeros' will be used

    weights_seed: (Union[int, Array]) – a 64- or 32-bit integer used as the value of the key.

    uniform_minval: see minval of jax.random.uniform

    uniform_maxval: see maxval of jax.random.uniform

    Returns
    -------
    function that takes shape=integer in input and returns array of shape = (shape, )

    """
    if weights_initializer not in ("zeros", "normal", "uniform"):
        log.warning(
            f"weights_initializer {weights_initializer} name not recognized, using default: 'zeros' instead"
        )
        weights_initializer = "zeros"

    if weights_initializer == "zeros":
        return jnp.zeros

    elif weights_initializer == "normal":
        rng = jax.random.PRNGKey(weights_seed)
        initializer = lambda shape: jax.random.normal(key=rng, shape=(shape,))
        return initializer

    elif weights_initializer == "uniform":
        rng = jax.random.PRNGKey(weights_seed)
        initializer = lambda shape: jax.random.uniform(
            key=rng, shape=(shape,), minval=uniform_minval, maxval=uniform_maxval
        )
        return initializer


@check_wminpdfset_is_montecarlo
def lhapdf_from_collected_weights(
    wminpdfset,
    mc_replicas_weight_minimization_fit,
    n_replicas,
    *,
    folder=None,
    set_name=None,
    errortype: str = "replicas",
):
    """
    TODO
    """

    original_pdf = pathlib.Path(lhapdf.paths()[-1]) / str(wminpdfset)
    if folder is None:
        # requested folder for the new LHAPDF to reside
        folder = ""

    if set_name is None:
        set_name = str(wminpdfset) + "_wmin"

    wm_pdf = pathlib.Path(folder) / set_name
    if not wm_pdf.exists():
        os.makedirs(wm_pdf)

    with open(original_pdf / f"{wminpdfset}.info", "r") as in_stream, open(
        wm_pdf / f"{set_name}.info", "w"
    ) as out_stream:
        for l in in_stream.readlines():
            if l.find("SetDesc:") >= 0:
                out_stream.write(f'SetDesc: "Weight-minimized {wminpdfset}"\n')
            elif l.find("NumMembers:") >= 0:
                out_stream.write(f"NumMembers: {n_replicas + 1}\n")
            elif l.find("ErrorType: replicas") >= 0:
                out_stream.write(f"ErrorType: {errortype}\n")
            else:
                out_stream.write(l)

    headers, grids = load_all_replicas(wminpdfset)
    replicas_df = rep_matrix(grids)

    # each wmin replica could in principle have its own wminpdfset replica basis
    for i, wmin_fit in enumerate(mc_replicas_weight_minimization_fit):
        # take care of different indexing. Central replica is at index 1
        wmin_basis_idx = wmin_fit.wmin_basis_idx + 1
        wmin_central_replica = wmin_fit.wmin_central_replica + 1
        optimised_wmin_weights = wmin_fit.optimised_wmin_weights

        rep0, replica = (
            replicas_df.loc[:, [int(wmin_central_replica)]],
            replicas_df.loc[:, wmin_basis_idx],
        )

        wm_replica = rep0.dot([1.0 - jnp.sum(optimised_wmin_weights)]) + replica.dot(
            optimised_wmin_weights
        )

        # for i, replica in tqdm(enumerate(result), total=len(weights)):
        wm_headers = f"PdfType: replica\nFormat: lhagrid1\nFromMCReplica: {i}\n"
        write_replica(i + 1, wm_pdf, wm_headers.encode("UTF-8"), wm_replica)

    # TODO
    # read pdf into validphys.core.PDF and generate central replica

    # pdf_obj = core.PDF(wm_pdf)
    # import IPython
    # IPython.embed()
    # generate_replica0(pdf_obj)
