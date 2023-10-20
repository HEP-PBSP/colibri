"""
TODO
"""
from super_net.app import providers, SuperNetApp
from wmin.config import WminConfig

import pathlib

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

    @property
    def argparser(self):
        parser = super().argparser

        parser.add_argument(
            '-o',
            '--output',
            nargs='?',
            default=None,
            help='Name of the output directory.',
        )

        return parser

    def get_commandline_arguments(self, cmdline=None):
        args = super().get_commandline_arguments(cmdline)
        if args['output'] is None:
            args['output'] = pathlib.Path(args['config_yml']).stem
        return args

def main():
    a = WminApp(name='wmin', providers=wmin_providers)
    a.main()


if __name__ == "__main__":
    main()
