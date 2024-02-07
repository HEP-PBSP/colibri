"""
api.py

This module contains the `reportengine` programmatic API, initialized with the
colibri providers, Config and Environment.

Example:
--------

Simple Usage:

>> from colibri.api import API
>> fig = API.plot_pdfs(pdf="NNPDF_nlo_as_0118", Q=100)
>> fig.show()

"""

import logging

from reportengine import api
from colibri.app import colibri_providers
from colibri.config import colibriConfig, Environment

log = logging.getLogger(__name__)

# API needed its own module, so that it can be used with any Matplotlib backend
# without breaking validphys.app
API = api.API(colibri_providers, colibriConfig, Environment)
