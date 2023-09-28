from validphys.pseudodata import make_level1_data
from validphys.covmats import dataset_t0_predictions


def experimental_commondata_tuple(data):
    """
    returns a tuple (validphys nodes should be immutable)
    of commondata instances with experimental central values

    Parameters
    ----------
    data: super_net.core.SuperNetDataGroupSpec

    Returns
    -------
    tuple
        tuple of validphys.coredata.CommonData instances
    """
    return tuple(data.load_commondata_instance())


def pseudodata_commondata_tuple(data, experimental_commondata_list, filterseed=1):
    """
    returns a tuple (validphys nodes should be immutable)
    of commondata instances with experimental central values
    fluctuated with random noise sampled from experimental
    covariance matrix

    Parameters
    ----------
    data: super_net.core.SuperNetDataGroupSpec

    experimental_commondata_tuple: tuple
        tuple of commondata with experimental central values

    filterseed: int, default is 1
        seed used for the sampling of random noise

    Returns
    -------
    tuple
        tuple of validphys.coredata.CommonData instances
    """

    index = data.data_index()
    dataset_order = [cd.setname for cd in experimental_commondata_list]
    pseudodata_list = make_level1_data(
        data, experimental_commondata_list, filterseed, index, sep_mult=True
    )
    pseudodata_list = sorted(
        pseudodata_list, key=lambda obj: dataset_order.index(obj.setname)
    )
    return tuple(pseudodata_list)


def closuretest_commondata_tuple(data, experimental_commondata_tuple, closure_test_pdf):
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

    closure_test_pdf: validphys.core.PDF
        PDF used to generate fake data

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
            cd.with_central_value(dataset_t0_predictions(ds, closure_test_pdf))
        )
    return tuple(fake_data)


def closuretest_pseudodata_commondata_tuple(data, closuretest_commondata_tuple, filterseed=1):
    """
    Like `pseudodata_commondata_tuple` but with closure test (fake-data) central values.

    Returns
    -------
    tuple
        tuple of validphys.coredata.CommonData instances
    """
    return pseudodata_commondata_tuple(data, closuretest_commondata_tuple, filterseed)