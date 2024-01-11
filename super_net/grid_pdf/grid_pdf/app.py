"""
grid_pdf.app.py

Author: Mark N. Costantini
Date: 15.11.2023
"""

from super_net.app import SuperNetApp
from grid_pdf.config import GridPdfConfig

import pathlib

grid_pdf_providers = [
    "grid_pdf.grid_pdf_model",
    "grid_pdf.grid_pdf_fit",
    "grid_pdf.grid_pdf_utils",
    "grid_pdf.grid_pdf_lhapdf",
    "grid_pdf.provider_aliases",
    "grid_pdf.utils",
]


class GridPdfApp(SuperNetApp):
    config_class = GridPdfConfig

    @property
    def argparser(self):
        """Parser arguments for grid_pdf app can be added here"""
        parser = super().argparser

        parser.add_argument(
            "-o",
            "--output",
            nargs="?",
            default=None,
            help="Name of the output directory.",
        )

        return parser

    def get_commandline_arguments(self, cmdline=None):
        """Get commandline arguments"""
        args = super().get_commandline_arguments(cmdline)
        if args["output"] is None:
            args["output"] = pathlib.Path(args["config_yml"]).stem
        return args


def main():
    a = GridPdfApp(name="grid_pdf", providers=providers + grid_pdf_providers)
    a.main()


if __name__ == "__main__":
    main()
