"""
wmin.app.py

The wmin app.
"""

from colibri.app import colibriApp
from wmin.config import WminConfig

wmin_providers = [
    "wmin.model",
    "wmin.utils",
]


class WminApp(colibriApp):
    config_class = WminConfig


def main():
    a = WminApp(name="wmin", providers=wmin_providers)
    a.main()


if __name__ == "__main__":
    main()
