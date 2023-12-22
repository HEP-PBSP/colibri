"""
grid_pdf.app.py

Author: Mark N. Costantini
Date: 15.11.2023
"""

from super_net.app import SuperNetApp, providers
from grid_pdf.config import GridPdfConfig

grid_pdf_providers = [
    *providers,
    "reportengine.report",
    "grid_pdf.grid_pdf_model",
    "grid_pdf.grid_pdf_fit",
    "grid_pdf.grid_pdf_utils",
    "grid_pdf.provider_aliases",
]


class GridPdfApp(SuperNetApp):
    config_class = GridPdfConfig


def main():
    a = GridPdfApp(name="grid_pdf", providers=grid_pdf_providers)
    a.main()


if __name__ == "__main__":
    main()
