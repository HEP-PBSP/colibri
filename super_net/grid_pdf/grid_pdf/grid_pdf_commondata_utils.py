import jax.numpy as jnp

from validphys import convolution

from super_net.constants import XGRID
from super_net.utils import FLAVOURS_ID_MAPPINGS

from super_net.theory_predictions import make_pred_dataset


def closure_test_pdf_grid_interpolated(closure_test_pdf, xgrids, Q0=1.65):
    """
    Computes the closure_test_pdf grid in the evolution basis and


    Parameters
    ----------
    closure_test_pdf: validphys.core.PDF

    xgrids: dict
        defines the reduced grid, keys are flavour names and values are x values.
        Each flavour needs to have a reduced grid assigned to in xgrids and
        all flavours need to have the same number of x values.

    Q0: float, default is 1.65

    Returns
    -------
    grid: jnp.array
        grid, is N_rep x N_fl x N_x
    """

    # Every flavour (even unused ones like photon) needs to have a reduced grid assigned to in xgrids
    # the flavour selection/mapping is then done by flavour_mapping (flavour_indices)

    reduced_xgrid = jnp.concatenate(
        [
            convolution.evolution.grid_values(
                closure_test_pdf, [fl], x_vals, [Q0]
            ).squeeze(-1)
            for fl, x_vals in xgrids.items()
        ],
        axis=1,
    )

    interpolated_xgrid = jnp.zeros((reduced_xgrid.shape[0], reduced_xgrid.shape[1], len(XGRID)))

    for rep_idx in range(reduced_xgrid.shape[0]):

        for fl_idx in range(reduced_xgrid.shape[1]):

                interpolated_xgrid = interpolated_xgrid.at[rep_idx, fl_idx, :].set(
                    jnp.interp(
                        jnp.array(XGRID),
                        jnp.array(xgrids[FLAVOURS_ID_MAPPINGS[fl_idx]]),
                        reduced_xgrid[rep_idx, fl_idx, :],
                    )
                )
    
    return interpolated_xgrid 
    


def grid_pdf_closuretest_commondata_tuple(
    data,
    experimental_commondata_tuple,
    closure_test_pdf_grid_interpolated,
    flavour_indices=None,
):
    """
    returns a tuple (validphys nodes should be immutable)
    of commondata instances with experimental central values
    replaced with theory predictions computed from a PDF `closure_test_pdf`
    and fktables corresponding to datasets within data

    Parameters
    ----------
    data: super_net.core.SuperNetDataGroupSpec

    experimental_commondata_tuple: tuple
        tuple of commondata with experimental central values

    closure_test_pdf_grid: jnp.array
        grid is of shape N_rep x N_fl x N_x

    Returns
    -------
    tuple
        tuple of validphys.coredata.CommonData instances
    """

    fake_data = []
    for cd, ds in zip(experimental_commondata_tuple, data.datasets):
        if cd.setname != ds.name:
            raise RuntimeError(f"commondata {cd} does not correspond to dataset {ds}")
        # replace central values with theory prediction from `closure_test_pdf`
        fake_data.append(
            cd.with_central_value(
                make_pred_dataset(ds, flavour_indices=flavour_indices)(
                    closure_test_pdf_grid_interpolated[0]
                )
            )
        )
    return tuple(fake_data)
