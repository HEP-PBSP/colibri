import pandas as pd

from validphys.core import DataGroupSpec
from validphys.pseudodata import make_level1_data


class SuperNetDataGroupSpec(DataGroupSpec):
    """
    Class inheriting from `validphys.core.DataGroupSpec`
    """

    def data_index(self):
        """
        index of data, needed by `make_replica`
        """
        tuples = []
        for ds in self.datasets:
            for i in ds.cuts.load():
                tp = ("test", ds.name, i)
                tuples.append(tp)
        return pd.MultiIndex.from_tuples(tuples, names=("group", "dataset", "id"))

    def load_pseudo_commondata(self, pseudo=False, filterseed=1):
        """
        Like `load_commondata_instance` but allowing for the generation
        of pseudodata trough make_level1_data function

        Parameters
        ----------
        pseudo : bool, default is False
                    when False, load_commondata_instance, i.e. the experimental
                    commondata is returned. Otherwise noise is added to the
                    central values of the commondata
        filterseed : int, default is 1
                    seed used for the sampling of random noise

        Returns
        -------
        list
            list of commondata
        """

        if not pseudo:
            return self.load_commondata_instance()

        else:
            index = self.data_index()
            experimental_data = self.load_commondata_instance()
            dataset_order = [cd.setname for cd in experimental_data]
            pseudodata = make_level1_data(
                self, experimental_data, filterseed, index, sep_mult=True
            )
            pseudodata = sorted(
                pseudodata, key=lambda obj: dataset_order.index(obj.setname)
            )
            return pseudodata
