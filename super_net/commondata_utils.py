from validphys.pseudodata import make_level1_data

from reportengine import collect


def pseudodata_commondata_tuple(data, experimental_commondata_tuple, filterseed=1):
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
    dataset_order = [cd.setname for cd in experimental_commondata_tuple]
    pseudodata_list = make_level1_data(
        data, experimental_commondata_tuple, filterseed, index, sep_mult=True
    )
    pseudodata_list = sorted(
        pseudodata_list, key=lambda obj: dataset_order.index(obj.setname)
    )
    return tuple(pseudodata_list)


def closuretest_pseudodata_commondata_tuple(
    data, closuretest_commondata_tuple, filterseed=1
):
    """
    Like `pseudodata_commondata_tuple` but with closure test (fake-data) central values.

    Returns
    -------
    tuple
        tuple of validphys.coredata.CommonData instances
    """
    return pseudodata_commondata_tuple(data, closuretest_commondata_tuple, filterseed)


"""
Collect over multiple random filterseeds so as to generate multiple commondata instances.
To be used in a Monte Carlo fit to experimental data.
"""
mc_replicas_pseudodata_commondata_tuple = collect(
    "pseudodata_commondata_tuple", ("pseudodata_replica_collector_helper",)
)

"""
Collect over multiple random filterseeds so as to generate multiple commondata instances.
To be used in a Monte Carlo closure test fit.
"""
mc_replicas_closuretest_pseudodata_commondata_tuple = collect(
    "closuretest_pseudodata_commondata_tuple",
    ("closure_test_replica_collector_helper",),
)
