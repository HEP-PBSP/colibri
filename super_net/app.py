"""
super_net.app.py

Author: Mark N. Costantini
Date: 11.11.2023
"""
from validphys.app import App
from super_net.config import SuperNetConfig

import pathlib

providers = [
    "reportengine.report",
    "super_net.theory_predictions",
    "super_net.loss_functions",
    "super_net.optax_optimizer",
    "super_net.data_batch",
    "super_net.utils",
    "super_net.commondata_utils",
    "super_net.closure_test.closure_test_estimators",
    "super_net.training_validation",
    "super_net.covmats",
    "super_net.plots_and_tables.plotting",
    "super_net.provider_aliases",
    "super_net.mc_fit",
]

from wmin.app import wmin_providers
from grid_pdf.app import grid_pdf_providers

class SuperNetApp(App):
    config_class = SuperNetConfig

    @property
    def argparser(self):
        parser = super().argparser

        parser.add_argument(
            '-o',
            '--output',
            nargs='?',
            default=None,
            help='Name of the output directory.',
        )

        return parser

    def get_commandline_arguments(self, cmdline=None):
        args = super().get_commandline_arguments(cmdline)
        if args['output'] is None:
            args['output'] = pathlib.Path(args['config_yml']).stem
        return args

def main():
    a = SuperNetApp(name="super_net", providers=providers+wmin_providers+grid_pdf_providers)
    a.main()


if __name__ == "__main__":
    main()
