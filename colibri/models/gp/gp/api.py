"""
api.py

This module contains the `reportengine` programmatic API, initialized with the
colibri providers, Config and Environment.

Example:
--------

Simple Usage:

>> from gp.api import API as gpAPI
>> fig = gpAPI.plot_pdfs(pdf="NNPDF_nlo_as_0118", Q=100)
>> fig.show()
"""

import logging

from reportengine import api
from colibri.app import colibri_providers
from gp.app import gp_providers
from gp.config import GpPdfConfig, Environment

log = logging.getLogger(__name__)

# API needed its own module, so that it can be used with any Matplotlib backend
# without breaking validphys.app
API = api.API(gp_providers + colibri_providers, GpPdfConfig, Environment)
