"""
grid_pdf.utils.py

Module containing util functions for grid PDF fits.

Author: Luca Mantani
Date: 18.12.2023
"""

import logging
from datetime import datetime
import jax
import jax.numpy as jnp

log = logging.getLogger(__name__)


def gridpdf_fit_name(set_name=None):
    if set_name:
        return set_name
    current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    return f"{current_time}_grid_fit"


def init_stacked_pdf_grid(
    length_stackedpdf,
    grid_initializer,
    replica_index,
):
    if grid_initializer["type"] == "zeros":
        return jnp.zeros(shape=length_stackedpdf)

    elif grid_initializer["type"] == "uniform":
        rng = jax.random.PRNGKey(replica_index)

        return jax.random.uniform(
            rng,
            shape=(length_stackedpdf,),
            minval=grid_initializer["minval"],
            maxval=grid_initializer["maxval"],
        )
