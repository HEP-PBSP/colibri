"""
grid_pdf.grid_pdf_utils.py

Module containing util functions for grid PDF fits.

Author: Luca Mantani
Date: 18.12.2023
"""

import logging
from datetime import datetime

log = logging.getLogger(__name__)


def gridpdf_fit_name(set_name=None):
    if set_name:
        return set_name
    current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    return f"{current_time}_grid_fit"
