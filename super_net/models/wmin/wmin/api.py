"""
api.py

This module contains the `reportengine` programmatic API, initialized with the
super_net providers, Config and Environment.

Example:
--------

Simple Usage:

>> from wmin.api import API
>> fig = API.plot_pdfs(pdf="NNPDF_nlo_as_0118", Q=100)
>> fig.show()

"""
import logging

from reportengine import api
from super_net.app import super_net_providers
from wmin.app import wmin_providers
from wmin.config import WminConfig, Environment

log = logging.getLogger(__name__)

# API needed its own module, so that it can be used with any Matplotlib backend
# without breaking validphys.app
API = api.API(wmin_providers + super_net_providers, WminConfig, Environment)
