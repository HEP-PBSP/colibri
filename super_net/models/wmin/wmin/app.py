"""
wmin.app.py

Author: Mark N. Costantini
Date: 11.11.2023
"""

from super_net.app import SuperNetApp
from wmin.config import WminConfig

import pathlib

wmin_providers = [
    "wmin.wmin_model",
]


class WminApp(SuperNetApp):
    config_class = WminConfig


def main():
    a = WminApp(name="wmin", providers=wmin_providers)
    a.main()


if __name__ == "__main__":
    main()
