"""
colibri.app.py

Module contains the main class for the colibri app.
"""

from validphys.app import App
from colibri.config import colibriConfig, Environment
import pathlib


colibri_providers = [
    "colibri.theory_predictions",
    "colibri.theory_penalties",
    "colibri.loss_functions",
    "colibri.mc_loss_functions",
    "colibri.optax_optimizer",
    "colibri.data_batch",
    "colibri.utils",
    "colibri.commondata_utils",
    "colibri.training_validation",
    "colibri.covmats",
    "colibri.provider_aliases",
    "colibri.mc_utils",
    "colibri.likelihood",
    "colibri.ultranest_fit",
    "colibri.blackjax_fit",
    "colibri.monte_carlo_fit",
    "colibri.analytic_fit",
    "colibri.pdf_model",
    "colibri.bayes_prior",
    "colibri.mc_initialisation",
    "colibri.export_results",
    "colibri.closure_test",
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
            "-f32",
            "--float32",
            action="store_true",
            help="Use float32 precision for the computation",
        )

        return parser

    def get_commandline_arguments(self, cmdline=None):
        """Get commandline arguments"""
        args = super().get_commandline_arguments(cmdline)
        if args["output"] is None:
            args["output"] = pathlib.Path(args["config_yml"]).stem
        return args


def main():
    a = colibriApp(name="colibri")
    a.main()


if __name__ == "__main__":
    main()
