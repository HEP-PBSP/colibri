"""
wmin.app.py

The wmin app.
"""

from super_net.app import SuperNetApp
from wmin.config import WminConfig

wmin_providers = [
    "wmin.model",
]


class WminApp(SuperNetApp):
    config_class = WminConfig


def main():
    a = WminApp(name="wmin", providers=wmin_providers)
    a.main()


if __name__ == "__main__":
    main()
