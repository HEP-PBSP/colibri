import os
import pathlib
import logging

import lhapdf
import numpy as np
import jax
import jax.numpy as jnp

from validphys.lhio import load_all_replicas, write_replica, rep_matrix
from tqdm import tqdm

from super_net.checks import check_wminpdfset_is_montecarlo

log = logging.getLogger(__name__)


@check_wminpdfset_is_montecarlo
def lhapdf_from_weights(
    wminpdfset,
    weights,
    *,
    folder:str= None,
    set_name:str = None,
    errortype:str = "replicas",
    wmin_basis_idxs: np.array,
    rep1_idxs: int
) -> pathlib.Path:
    """
    TODO

    Parameters
    ----------
    wminpdfset: validphys.core.PDF

    weights: 
        np.array of size monte_carlo_replicas x n_replicas_wmin.

    folder : str, default is None

    set_name : str, default is None

    errortype: str = "replicas"

    wmin_basis_idxs: np.array

    rep1_idxs: int

    Returns
    -------

    Weights is 
    """

    mc_replicas, nr_wmin_rep = np.shape(weights)
    
    original_pdf = pathlib.Path(lhapdf.paths()[-1]) / str(wminpdfset)
    if folder is None:
        # requested folder for the new LHAPDF to reside
        folder = ""
    
    if set_name is None:
        set_name = str(wminpdfset)+"_wmin"

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
                out_stream.write(f"NumMembers: {mc_replicas + 1}\n")
            elif l.find("ErrorType: replicas") >= 0:
                out_stream.write(f"ErrorType: {errortype}\n")
            else:
                out_stream.write(l)

    headers, grids = load_all_replicas(wminpdfset)
    replicas_df = rep_matrix(grids)

    # each wmin replica could in principle have its own wminpdfset replica basis
    for i, (wmin_basis_idx, rep1_idx, weight) in enumerate(zip(wmin_basis_idxs, rep1_idxs, weights)):
        
        # take care of different indexing. Central replicas is at index 1
        wmin_basis_idx = np.array(wmin_basis_idx)+1

        rep0, replica = replicas_df.loc[:,[int(rep1_idx)]], replicas_df.loc[:,np.delete(wmin_basis_idx, int(rep1_idx))]

        wm_replica = (rep0.dot([1. - np.sum(weight)]) + replica.dot(weight))

        # for i, replica in tqdm(enumerate(result), total=len(weights)):
        wm_headers = f"PdfType: replica\nFormat: lhagrid1\nFromMCReplica: {i}\n"
        write_replica(i + 1, wm_pdf, wm_headers.encode("UTF-8"), wm_replica)
        

    return wm_pdf


def weights_initializer_provider(weights_initializer='zeros', weights_seed=0xABCDEF, uniform_minval=-0.1, uniform_maxval=0.1):
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
    if weights_initializer not in ('zeros', 'normal', 'uniform'):
        log.warning(f"weights_initializer {weights_initializer} name not recognized, using default: 'zeros' instead")
        weights_initializer = 'zeros'

    if weights_initializer=='zeros':
        return jnp.zeros
    
    elif weights_initializer=='normal':
        rng = jax.random.PRNGKey(weights_seed)
        initializer = lambda shape : jax.random.normal(key=rng, shape=(shape,))
        return initializer
    
    elif weights_initializer=='uniform':
        rng = jax.random.PRNGKey(weights_seed)
        initializer = lambda shape: jax.random.uniform(key=rng, shape=(shape,), minval=uniform_minval, maxval=uniform_maxval)
        return initializer
