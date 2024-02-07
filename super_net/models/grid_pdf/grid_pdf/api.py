"""
api.py

This module contains the `reportengine` programmatic API, initialized with the
super_net providers, Config and Environment.

Example:
--------

Simple Usage:

>> from grid_pdf.api import API
>> fig = API.plot_pdfs(pdf="NNPDF_nlo_as_0118", Q=100)
>> fig.show()
"""

import logging

from reportengine import api
from super_net.app import super_net_providers
from grid_pdf.app import grid_pdf_providers
from grid_pdf.config import GridPdfConfig, Environment

log = logging.getLogger(__name__)

# API needed its own module, so that it can be used with any Matplotlib backend
# without breaking validphys.app
API = api.API(grid_pdf_providers + super_net_providers, GridPdfConfig, Environment)
