"""
super_net.app.py

Author: Mark N. Costantini
Date: 11.11.2023
"""

from validphys.app import App
from super_net.config import SuperNetConfig, Environment


super_net_providers = [
    "super_net.theory_predictions",
    "super_net.loss_functions",
    "super_net.mc_loss_functions",
    "super_net.optax_optimizer",
    "super_net.data_batch",
    "super_net.utils",
    "super_net.commondata_utils",
    "super_net.training_validation",
    "super_net.covmats",
    "super_net.plots_and_tables.plotting",
    "super_net.provider_aliases",
    "super_net.mc_utils",
    "reportengine.report",
]


class SuperNetApp(App):
    config_class = SuperNetConfig
    environment_class = Environment

    def __init__(self, name="super_net", providers=[]):
        super().__init__(name, super_net_providers + providers)

    @property
    def argparser(self):
        """Parser arguments for grid_pdf app can be added here"""
        parser = super().argparser

        parser.add_argument(
            "-rep", "--replica_index", help="MC replica number", type=int, default=None
        )

        return parser


def main():
    a = SuperNetApp(name="super_net")
    a.main()


if __name__ == "__main__":
    main()
