"""
colibri.compute_svd.py
"""

import logging

import numpy as np

from validphys import convolution
from validphys.fkparser import load_fktable
from colibri.theory_predictions import fktable_xgrid_indices

log = logging.getLogger(__name__)

# Size of the evolution-basis flavour space that every DIS FK table gets
# zero-padded onto (same basis as colibri.constants.FLAVOUR_TO_ID_MAPPING).
N_FLAVOURS = len(convolution.FK_FLAVOURS)


def _pad_dis_fktable(fk, FIT_XGRID):
    """
    Zero-pads a (cut-applied) DIS FKTableData onto the common
    (N_FLAVOURS, len(FIT_XGRID)) basis.

    Parameters
    ----------
    fk : validphys.coredata.FKTableData
        A non-hadronic FK table, with cuts already applied.

    FIT_XGRID : array-like
        The common x-grid entering the fit.

    Returns
    -------
    np.ndarray
        Array of shape (Ndat, N_FLAVOURS * len(FIT_XGRID)).
    """
    fk_arr = np.asarray(fk.get_np_fktable())  # (Ndat, n_present_fl, Nx_local)
    n_dat = fk_arr.shape[0]

    lumi_indices = np.asarray(fk.luminosity_mapping)
    x_indices = np.asarray(fktable_xgrid_indices(fk, FIT_XGRID))

    padded = np.zeros((n_dat, N_FLAVOURS, len(FIT_XGRID)))
    padded[:, lumi_indices[:, None], x_indices[None, :]] = fk_arr

    return padded.reshape(n_dat, -1)


def compute_svd(
    data,
    FIT_XGRID,
    output_path,
    flavour_indices=None,
    max_singular_values=None,
):
    """
    Builds a single combined matrix out of every DIS FK table entering
    ``data`` (all datasets in the runcard, stacked together), padded onto
    the full N_FLAVOURS=14 evolution-basis flavour space, and computes its
    singular value spectrum. Hadronic FK tables are skipped, with a
    warning listing which ones. Stores a single CSV file under
    ``<output_path>/fk_svd/dis_fk_svd.csv`` -- no plot is produced by this
    action; plot the CSV separately with your own script (see
    plot_svd.py) for full control over the x-axis, scale, etc.

    Parameters
    ----------
    data : validphys.core.DataGroupSpec
        The data entering the computation, as specified by
        ``dataset_inputs``, ``theoryid`` and ``use_cuts`` in the runcard.

    FIT_XGRID : np.ndarray
        Common fit x-grid, produced automatically by
        ``colibriConfig.produce_FIT_XGRID`` -- no need to specify it in the
        runcard.

    output_path : pathlib.Path
        Colibri output folder. Automatically provided by
        ``colibri.config.Environment``.

    flavour_indices : list, default is None
        Optional list of evolution-basis flavour indices (see
        ``colibri.constants.FLAVOUR_TO_ID_MAPPING``) to restrict to before
        computing the SVD. Wired up automatically if you set
        ``flavour_mapping`` in the runcard (see
        ``colibriConfig.produce_flavour_indices``). If None (default), all
        N_FLAVOURS=14 flavours are kept, i.e. the full padded basis is
        used (this is what gives e.g. 700 = 14 * 50 columns/rows in the
        CSV rather than a smaller auto-detected subset).

    max_singular_values : int, default is None
        If given and smaller than the number of columns of the combined
        matrix, only the largest ``max_singular_values`` singular values
        are written to the CSV (the matrix is still fully decomposed to
        compute rank/condition number). Can be set directly in the
        runcard.

    Returns
    -------
    np.ndarray or None
        The (possibly capped) array of singular values (decreasing
        order), or None if no DIS FK tables were found.
    """
    svd_folder = output_path / "fk_svd"
    svd_folder.mkdir(exist_ok=True)

    blocks = []
    skipped = []

    for ds in data.datasets:
        for i, fkspec in enumerate(ds.fkspecs):
            fk = load_fktable(fkspec).with_cuts(ds.cuts)

            if fk.hadronic:
                skipped.append(f"{ds.name}_fk{i}")
                continue

            block = _pad_dis_fktable(fk, FIT_XGRID)
            log.info(f"{ds.name}_fk{i}: DIS FK table padded to shape {block.shape}")
            blocks.append(block)

    if skipped:
        log.warning(
            f"Skipped {len(skipped)} hadronic (bilinear) FK table(s), not "
            f"included in the SVD since they are not linear in the PDF: {skipped}"
        )

    if not blocks:
        log.error("No DIS FK tables found in the data -- nothing to do.")
        return None

    fk_matrix = np.concatenate(blocks, axis=0)

    if flavour_indices is not None:
        n_x = len(FIT_XGRID)
        fk_matrix = fk_matrix.reshape(fk_matrix.shape[0], N_FLAVOURS, n_x)
        fk_matrix = fk_matrix[:, np.asarray(flavour_indices), :].reshape(
            fk_matrix.shape[0], -1
        )

    n_cols = fk_matrix.shape[1]

    log.info(
        f"Combined DIS FK matrix built from {len(blocks)} FK table(s) "
        f"(datasets: {sorted({ds.name for ds in data.datasets})}), "
        f"total shape {fk_matrix.shape} (N_FLAVOURS * Nx = {n_cols})"
    )

    log.info("Computing SVD of the combined DIS FK matrix...")
    singular_values = np.linalg.svd(fk_matrix, compute_uv=False)

    rank = int(np.sum(singular_values > 1e-12 * singular_values[0]))
    cond = (
        singular_values[0] / singular_values[-1] if singular_values[-1] > 0 else np.inf
    )
    log.info(
        f"Combined DIS FK matrix: shape = {fk_matrix.shape}, "
        f"numerical rank = {rank}/{len(singular_values)}, "
        f"condition number = {cond:.3e}"
    )

    cutoff = n_cols if max_singular_values is None else min(max_singular_values, n_cols)
    stored_values = singular_values[:cutoff]

    csv_path = svd_folder / "dis_fk_svd.csv"
    np.savetxt(
        csv_path,
        np.column_stack([np.arange(len(stored_values)), stored_values]),
        header="index,singular_value",
        delimiter=",",
        comments="",
    )
    log.info(f"Singular values ({len(stored_values)} rows) saved to {csv_path}")

    return stored_values
