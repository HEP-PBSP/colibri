"""
wmin.config.py

Config module of wmin

"""

from super_net.config import SuperNetConfig, Environment


class Environment(Environment):
    pass


class WminConfig(SuperNetConfig):
    """
    WminConfig class Inherits from super_net.config.SuperNetConfig
    """

    def parse_prior_settings(self, settings):
        # Currently, only prior is uniform with max/min val
        if settings['type'] == 'uniform_parameter_prior':
            # Check if max and min vals are defined, if not set them to defaults
            # of -1 and 1.
            if 'min_val' not in settings.keys():
                settings['min_val'] = -1
            if 'max_val' not in settings.keys():
                settings['max_val'] = 1

        return settings
