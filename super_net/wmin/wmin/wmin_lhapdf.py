"""
TODO
"""
import os
import pathlib
import json

import jax.numpy as jnp

import lhapdf

from validphys.lhio import (
    load_all_replicas,
    write_replica,
    rep_matrix,
)

from wmin.checks import check_wminpdfset_is_montecarlo


@check_wminpdfset_is_montecarlo
def lhapdf_from_collected_weights(
    wminpdfset,
    mc_replicas_weight_minimization_fit,
    n_replicas,
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

    replica_weights = []
    # each wmin replica could in principle have its own wminpdfset replica basis
    for i, wmin_fit in enumerate(mc_replicas_weight_minimization_fit):
        # take care of different indexing. Central replica is at index 1
        wmin_basis_idx = wmin_fit.wmin_basis_idx + 1
        wmin_central_replica = wmin_fit.wmin_central_replica + 1
        optimised_wmin_weights = wmin_fit.optimised_wmin_weights
        
        replica_weights.append(optimised_wmin_weights.tolist())

        wmin_centr_rep, replica = (
            replicas_df.loc[:, [int(wmin_central_replica)]],
            replicas_df.loc[:, wmin_basis_idx],
        )

        wm_replica = wmin_centr_rep.dot(
            [1.0 - jnp.sum(optimised_wmin_weights)]
        ) + replica.dot(optimised_wmin_weights)

        # for i, replica in tqdm(enumerate(result), total=len(weights)):
        wm_headers = f"PdfType: replica\nFormat: lhagrid1\nFromMCReplica: {i}\n"
        write_replica(i + 1, wm_pdf, wm_headers.encode("UTF-8"), wm_replica)

    # write optimized wmin weights to json file
    monte_carlo_result_set_name = str(wm_pdf) + "_monte_carlo_results"
    monte_carlo_res = pathlib.Path(folder) / monte_carlo_result_set_name
    if not monte_carlo_res.exists():
        os.makedirs(monte_carlo_res)

    # save ultranest results to json file
    json_dump = json.dumps(replica_weights)

    with open(monte_carlo_res/"monte_carlo_results.json", "w") as json_file:
        json.dump(json_dump, json_file)


    # TODO
    # read pdf into validphys.core.PDF and generate central replica

    # pdf_obj = core.PDF(wm_pdf)
    # import IPython
    # IPython.embed()
    # generate_replica0(pdf_obj)

class NumpyEncoder(json.JSONEncoder):
    """ 
    Special json encoder for numpy types 
    see: https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable
    """
    
    def default(self, obj):
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


@check_wminpdfset_is_montecarlo
def lhapdf_wmin_and_ultranest_result(
    wminpdfset,
    weight_minimization_ultranest,
    n_wmin_posterior_samples,
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
                out_stream.write(f"NumMembers: {n_wmin_posterior_samples + 1}\n")
            elif l.find("ErrorType: replicas") >= 0:
                out_stream.write(f"ErrorType: {errortype}\n")
            else:
                out_stream.write(l)

    headers, grids = load_all_replicas(wminpdfset)
    replicas_df = rep_matrix(grids)
        
    # take care of different indexing. Central replica is at index 1
    wmin_basis_idx = weight_minimization_ultranest.wmin_basis_idx + 1
    wmin_central_replica = weight_minimization_ultranest.wmin_central_replica + 1
    optimised_wmin_weights = weight_minimization_ultranest.optimised_wmin_weights

    for i, wmin_weight in enumerate(optimised_wmin_weights):
        wmin_centr_rep, replica = (
            replicas_df.loc[:, [int(wmin_central_replica)]],
            replicas_df.loc[:, wmin_basis_idx],
        )
        
        wm_replica = wmin_centr_rep.dot(
            [1.0 - jnp.sum(wmin_weight)]
        ) + replica.dot(wmin_weight)

        # for i, replica in tqdm(enumerate(result), total=len(weights)):
        wm_headers = f"PdfType: replica\nFormat: lhagrid1\nFromMCReplica: {i}\n"
        write_replica(i + 1, wm_pdf, wm_headers.encode("UTF-8"), wm_replica)


    # write ultranest result to json file
    ultranest_result_set_name = str(wm_pdf) + "_ultranest_results"
    ultranest_res = pathlib.Path(folder) / ultranest_result_set_name
    if not ultranest_res.exists():
        os.makedirs(ultranest_res)

    # save ultranest results to json file
    json_dump = json.dumps(weight_minimization_ultranest.ultranest_result, cls=NumpyEncoder)

    with open(ultranest_res/"ultranest_results.json", "w") as json_file:
        json.dump(json_dump, json_file)

    # note: to read json file:
    # with open("ultranest_results.json", "r") as file:
    #   data = json.loads(json.load(file))
    
        
