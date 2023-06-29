import os
import pathlib

import lhapdf
import numpy as np

from validphys.lhio import load_all_replicas, write_replica, rep_matrix
from tqdm import tqdm

from super_net.checks import check_wminpdfset_is_montecarlo


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
