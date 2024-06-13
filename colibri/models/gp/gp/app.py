"""
gp.app.py

The Gaussian Process app.
"""

from colibri.app import colibriApp
from gp.config import GPConfig


gp_providers = [
    "gp.model",
    "gp.utils",
]


class GridPdfApp(colibriApp):
    config_class = GPConfig


def main():
    a = GridPdfApp(name="col_gp", providers=gp_providers)
    a.main()


if __name__ == "__main__":
    main()
