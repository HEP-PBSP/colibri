"""
wmin.app.py

Author: Mark N. Costantini
Date: 11.11.2023
"""

from super_net.app import SuperNetApp
from wmin.config import WminConfig

import pathlib

wmin_providers = [
    "wmin.wmin_fit",
    "wmin.wmin_model",
    "wmin.wmin_utils",
    "wmin.wmin_lhapdf",
    "wmin.wmin_loss_functions",
]


class WminApp(SuperNetApp):
    config_class = WminConfig


def main():
    a = WminApp(name="wmin", providers=wmin_providers)
    a.main()


if __name__ == "__main__":
    main()
