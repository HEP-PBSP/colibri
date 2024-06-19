"""
grid_pdf.app.py

The grid_pdf app.
"""

from colibri.app import colibriApp
from grid_pdf.config import GridPdfConfig


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
