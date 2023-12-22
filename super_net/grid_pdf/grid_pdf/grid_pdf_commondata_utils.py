from super_net.theory_predictions import make_pred_dataset


def closure_test_pdf_grid_interpolated(closure_test_pdf, Q0=1.65):
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

    'TODO'


def grid_pdf_closuretest_commondata_tuple(
    data, experimental_commondata_tuple, closure_test_pdf_grid_interpolated, flavour_indices=None
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
