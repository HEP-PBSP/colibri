"""
wmin.app.py

Author: Mark N. Costantini
Date: 11.11.2023
"""

from colibri.app import colibriApp
from wmin.config import WminConfig

import pathlib

wmin_providers = [
    "wmin.model",
]


class WminApp(colibriApp):
    config_class = WminConfig


def main():
    a = WminApp(name="wmin", providers=wmin_providers)
    a.main()


if __name__ == "__main__":
    main()
