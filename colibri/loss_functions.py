"""
colibri.loss_functions.py

This module provides the functions necessary for the computation of the chi2.

Author: Mark N. Costantini
Date: 11.11.2023
"""

import jax.numpy as jnp


def chi2(central_values, predictions, inv_covmat):
    """ """
    diff = predictions - central_values

    loss = jnp.einsum("i,ij,j", diff, inv_covmat, diff)

    return loss


def make_pos_penalty(
    alpha,
    lambda_positivity,
    _penalty_posdata,
):
    """ """

    def pos_penalty(pdf, pos_fk_tables):
        return jnp.sum(_penalty_posdata(pdf, alpha, lambda_positivity, pos_fk_tables))

    return pos_penalty
