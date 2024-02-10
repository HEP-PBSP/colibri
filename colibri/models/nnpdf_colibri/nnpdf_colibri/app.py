"""
nnpdf_colibri.app.py

The nnpdf_colibri app.
"""

from colibri.app import colibriApp
from nnpdf_colibri.config import NNPDFColibriConfig


nnpdf_colibri_providers = [
    "nnpdf_colibri.model",
]


class NNPDFColibriApp(colibriApp):
    config_class = NNPDFColibriConfig


def main():
    a = NNPDFColibriApp(name="nnpdf_colibri", providers=nnpdf_colibri_providers)
    a.main()


if __name__ == "__main__":
    main()
