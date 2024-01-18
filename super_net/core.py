import pandas as pd

from validphys.core import DataGroupSpec
from validphys.pseudodata import make_level1_data
from validphys.covmats import dataset_t0_predictions


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

    def load_pseudo_commondata(
        self, pseudodata=False, filterseed=1, closure_test_pdf=None, fakedata=False
    ):
        """
        Like `load_commondata_instance` but allowing for the generation
        of pseudodata trough make_level1_data function

        Parameters
        ----------
        pseudodata: bool, default is False
                    when False, load_commondata_instance, i.e. the experimental
                    commondata is returned. Otherwise noise is added to the
                    central values of the commondata
        filterseed: int, default is 1
                    seed used for the sampling of random noise
        closure_test_pdf: validphys.core.PDF, default is None

        fakedata: bool, default is None
                whether to use fakedata in the fit

        Returns
        -------
        list
            list of commondata
        """

        # closure test data (fakedata)
        if fakedata:
            experimental_data = self.load_commondata_instance()
            fake_data = []
            for cd, ds in zip(experimental_data, self.datasets):
                if cd.setname != ds.name:
                    raise RuntimeError(
                        f"commondata {cd} does not correspond to dataset {ds}"
                    )
                # replace central values with theory prediction from `closure_test_pdf`
                fake_data.append(
                    cd.with_central_value(dataset_t0_predictions(ds, closure_test_pdf))
                )

            # level1 closure test, TODO
            return fake_data

        if pseudodata:
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

        return self.load_commondata_instance()
