from validphys.config import Config, Environment
from validphys import covmats

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
