import jax

from validphys.config import Config, Environment
from validphys import covmats
from validphys.covmats import dataset_t0_predictions

from reportengine.configparser import explicit_node

from super_net import commondata_utils
from super_net.core import SuperNetDataGroupSpec



class Environment(Environment):
    pass


class SuperNetConfig(Config):
    """
    Config class inherits from validphys
    Config class
    """

    def produce_example(self):
        return "example"

    def produce_data(
        self,
        data_input,
        *,
        group_name="data",
    ):
        """A set of datasets where correlated systematics are taken
        into account
        """
        datasets = []
        for dsinp in data_input:
            with self.set_context(ns=self._curr_ns.new_child({"dataset_input": dsinp})):
                datasets.append(self.parse_from_(None, "dataset", write=False)[1])

        return SuperNetDataGroupSpec(
            name=group_name, datasets=datasets, dsinputs=data_input
        )

    def produce_experimental_commondata_tuple(self, data):
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
    
    def produce_closuretest_commondata_tuple(self, data, experimental_commondata_tuple, closure_test_pdf):
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

    @explicit_node
    def produce_commondata_tuple(
        self, pseudodata=False, fakedata=False
    ):
        """ 
        Produces a commondata tuple node in the reportengine dag
        according to some options
        """

        if pseudodata and fakedata:
            # closure test pseudodata
            return commondata_utils.closuretest_pseudodata_commondata_tuple

        elif fakedata:
            # closure test fake-data
            return self.produce_closuretest_commondata_tuple
        
        elif pseudodata:
            # experimental central values + random noise from covmat
            return commondata_utils.pseudodata_commondata_tuple
        
        else:
            return self.produce_experimental_commondata_tuple

    def produce_mc_replica_seeds(self, monte_carlo_replicas=1, monte_carlo_replica_seed=1):
        """
        Generate a tuple of random seeds using jax.random.PRNGKey

        Parameters
        ----------
        monte_carlo_replicas: int
            number of monte carlo replicas
        
        monte_carlo_replica_seed: int
            seed used to initialize jax random generator
        
        Returns
        -------
        tuple
        """
        rng = jax.random.PRNGKey(monte_carlo_replica_seed)
        seeds = []
        for _ in range(monte_carlo_replicas):
            seeds.append(int(rng[0]))
            key, rng = jax.random.split(rng)
        return tuple(seeds)

    def produce_pseudodata_replica_collector_helper(self, data, experimental_commondata_tuple, mc_replica_seeds=[]):
        """
        Helper allowing commondata_utils.pseudodata_commondata_tuple to collect over different
        monte carlo seeds
        """
        res = []
        for seed in mc_replica_seeds:
            res.append({"data": data, "experimental_commondata_tuple":experimental_commondata_tuple, "filterseed":seed} )
        return res
    
    def produce_closure_test_replica_collector_helper(self, data, closuretest_commondata_tuple, mc_replica_seeds=[]):
        """
        Helper allowing commondata_utils.closuretest_pseudodata_commondata_tuple to collect over different
        monte carlo seeds
        """
        res = []
        for seed in mc_replica_seeds:
            res.append({"data": data, "closuretest_commondata_tuple":closuretest_commondata_tuple, "filterseed":seed} )
        return res


    def produce_dataset_inputs_t0_predictions(self, data, t0set, use_t0):
        """
        produce t0 predictions for all datasets in data
        """

        if not use_t0:
            raise (
                f"use_t0 needs to be set to True so that dataset_inputs_t0_predictions can be generated"
            )
        t0_predictions = []
        for dataset in data.datasets:
            t0_predictions.append(covmats.dataset_t0_predictions(dataset, t0set))
        return t0_predictions

    def parse_wminpdfset(self, name):
        """PDF set used to generate the weight minimization grid"""
        return self.parse_pdf(name)

    def parse_closure_test_pdf(self, name):
        """PDF set used to generate fakedata"""
        return self.parse_pdf(name)
