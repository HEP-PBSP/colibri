"""
grid_pdf.app.py

Author: Mark N. Costantini
Date: 15.11.2023
"""

from colibri.app import colibriApp
from grid_pdf.config import GridPdfConfig

import pathlib

grid_pdf_providers = [
    "grid_pdf.model",
    "grid_pdf.utils",
]


class GridPdfApp(colibriApp):
    config_class = GridPdfConfig


def main():
    a = GridPdfApp(name="grid_pdf", providers=grid_pdf_providers)
    a.main()


if __name__ == "__main__":
    main()
