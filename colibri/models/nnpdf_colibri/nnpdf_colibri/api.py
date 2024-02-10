"""
api.py

This module contains the `reportengine` programmatic API, initialized with the
colibri providers, Config and Environment.

Example:
--------

Simple Usage:

>> from nnpdf_colibri.api import API
>> fig = API.plot_pdfs(pdf="NNPDF_nlo_as_0118", Q=100)
>> fig.show()
"""

import logging

from reportengine import api
from colibri.app import colibri_providers
from nnpdf_colibri.app import nnpdf_colibri_providers
from nnpdf_colibri.config import NNPDFColibriConfig, Environment

log = logging.getLogger(__name__)

# API needed its own module, so that it can be used with any Matplotlib backend
# without breaking validphys.app
API = api.API(
    nnpdf_colibri_providers + colibri_providers, NNPDFColibriConfig, Environment
)
