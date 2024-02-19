"""
colibri.app.py

Author: Mark N. Costantini
Date: 11.11.2023
"""

from validphys.app import App
from colibri.config import colibriConfig, Environment
import pathlib

import jax
import logging

log = logging.getLogger(__name__)


colibri_providers = [
    "colibri.theory_predictions",
    "colibri.loss_functions",
    "colibri.mc_loss_functions",
    "colibri.optax_optimizer",
    "colibri.data_batch",
    "colibri.utils",
    "colibri.commondata_utils",
    "colibri.training_validation",
    "colibri.covmats",
    "colibri.plots_and_tables.plotting",
    "colibri.provider_aliases",
    "colibri.mc_utils",
    "colibri.ultranest_fit",
    "colibri.monte_carlo_fit",
    "colibri.analytic_fit",
    "colibri.pdf_model",
    "reportengine.report",
]


class colibriApp(App):
    config_class = colibriConfig
    environment_class = Environment

    def __init__(self, name="colibri", providers=[]):
        super().__init__(name, colibri_providers + providers)

    @property
    def argparser(self):
        """Parser arguments for grid_pdf app can be added here"""
        parser = super().argparser

        parser.add_argument(
            "-rep", "--replica_index", help="MC replica number", type=int, default=None
        )

        parser.add_argument(
            "--trval_index",
            help="Training/Validation seed used to perform the random split",
            type=int,
            default=0,
        )

        parser.add_argument(
            "-o",
            "--output",
            nargs="?",
            default=None,
            help="Name of the output directory.",
        )

        parser.add_argument(
            "--global_double_precision",
            "-gdp",
            action="store_true",
            help="Use double precision globally",
            default=True,
        )
        parser.add_argument(
            "--no-global_double_precision",
            "-ngdp",
            dest="global_double_precision",
            action="store_false",
        )

        return parser

    def get_commandline_arguments(self, cmdline=None):
        """Get commandline arguments"""
        args = super().get_commandline_arguments(cmdline)
        if args["output"] is None:
            args["output"] = pathlib.Path(args["config_yml"]).stem

        if args["global_double_precision"]:
            jax.config.update("jax_enable_x64", True)
        else:
            log.warning(
                "Using single precision globally, this may lead to numerical instability."
            )

        return args


def main():
    a = colibriApp(name="colibri")
    a.main()


if __name__ == "__main__":
    main()
