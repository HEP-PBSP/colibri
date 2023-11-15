"""
ppdf.app.py

Author: Mark N. Costantini
Date: 15.11.2023
"""

from super_net.app import SuperNetApp, providers
from ppdf.config import PpdfConfig

ppdf_providers = [
    *providers,
    "reportengine.report", 
    "ppdf.ppdf_model",
]


class PpdfApp(SuperNetApp):
    config_class = PpdfConfig


def main():
    a = PpdfApp(name="ppdf", providers=ppdf_providers)
    a.main()


if __name__ == "__main__":
    main()
