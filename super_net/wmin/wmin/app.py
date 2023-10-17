"""
TODO
"""
from super_net.app import SuperNetApp, providers
from wmin.config import WminConfig

wmin_providers = [
    *providers,
    "reportengine.report",
    "wmin.wmin_fit",
    "wmin.wmin_model",
    "wmin.wmin_utils",
    "wmin.wmin_lhapdf"
]


class WminApp(SuperNetApp):
    config_class = WminConfig


def main():
    a = WminApp(name="wmin", providers=wmin_providers)
    a.main()


if __name__ == "__main__":
    main()
