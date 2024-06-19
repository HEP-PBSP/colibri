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


class GpPdfApp(colibriApp):
    config_class = GPConfig


def main():
    a = GpPdfApp(name="gp", providers=gp_providers)
    a.main()


if __name__ == "__main__":
    main()
