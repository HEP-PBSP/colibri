"""
colibri.forward_map.py

Forward maps: parameters → theory predictions.

A ``ForwardMap`` implements the final stage of the fit pipeline, turning the
fit parameter vector into theory predictions that can be compared with
data in the likelihood. It will also also return the PDF values on the fit x-grid,
which is sometimes needed for computing penalties.


Design choice: fixed call signature
-----------------------------------
The log-likelihood calls every forward map with the same fixed signature::

    (pdf_grid_func, fk_tables, params) -> predictions, pdf

Parameter convention
--------------------
``params`` is a 1-D array containing *all* fit parameters. In colibri we allow
for "extra" fit parameters beyond the PDF model parameters (e.g. nuisance-like factors,
or parameters of a custom prediction function).

By convention:

``params[:self.n_pdf_params]`` are PDF parameters consumed by ``pdf_grid_func``;
any remaining entries are "extra" parameters interpreted by the chosen
  ``ForwardMap`` implementation.

Example - fitting a normalisation factor on top of the PDF
----------------------------------------------------------
::

    class NormForwardMap(ForwardMap):
        def __init__(self, pred_func, n_pdf_params: int):
            super().__init__(n_pdf_params)
            self._pred_func = pred_func

        def __call__(self, pdf_grid_func, fk_tables, params):
            pdf = pdf_grid_func(params[: self.n_pdf_params])
            norm = params[self.n_pdf_params]            # first extra parameter
            return norm * self._pred_func(pdf, fk_tables), pdf

Example - fixed PDF, fitting only extra parameters
---------------------------------------------------
::

    class FixedPDFForwardMap(ForwardMap):
        def __init__(self, pred_func, fixed_pdf, fk_tables, n_pdf_params: int = 0):
            super().__init__(n_pdf_params)
            self._pred_func = pred_func
            self.fixed_pdf = fixed_pdf
            self._fixed_pred = self._pred_func(fixed_pdf, fk_tables)

        def __call__(self, pdf_grid_func, fk_tables, params):
            scale = params[0]
            return scale * self._fixed_pred, self.fixed_pdf
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

import jax.numpy as jnp


class ForwardMap(ABC):
    """Abstract base class for forward maps.

    A forward map turns fit parameters into theory predictions that can be
    compared with experimental data inside the likelihood.

    All forward maps share the same call signature:

        ``(pdf_grid_func, fk_tables, params) -> predictions``

    Notes
    -----
    The split point between PDF parameters and "extra" parameters is owned
    by the forward map via ``self.n_pdf_params``.
    """

    def __init__(self, n_pdf_params: int):

        self.n_pdf_params = n_pdf_params

    @abstractmethod
    def __call__(
        self,
        pdf_grid_func: Callable[[jnp.ndarray], jnp.ndarray],
        fk_tables: Any,
        params: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute theory predictions from fit parameters.

        Parameters
        ----------
        pdf_grid_func : callable
            Callable that evaluates PDF values on the fit x-grid from the PDF
            parameters.

            Expected call signature:
                ``pdf = pdf_grid_func(pdf_params)``
            with ``pdf`` shaped ``(N_fl, N_x)``.

        fk_tables : jnp.ndarray
            Fast-kernel tables needed by the prediction function.

        params : jnp.ndarray
            1-D array containing all fit parameters. By convention:
              * ``params[:self.n_pdf_params]`` are PDF parameters
              * the remaining entries are extra parameters interpreted by the
                specific ``ForwardMap`` implementation.

        Returns
        -------
        jnp.ndarray
            Theory predictions (1-D array with one entry per data point).
        jnp.ndarray
            The PDF values (2-D array with shape (N_fl, N_x)).

        """
        raise NotImplementedError


class FKTableForwardMap(ForwardMap):
    """Default forward map: params → PDF → FK-table convolution.

    This is the standard pipeline used in colibri PDF fits.
    """

    def __init__(
        self, pred_func: Callable[[jnp.ndarray, Any], jnp.ndarray], n_pdf_params: int
    ):
        super().__init__(n_pdf_params)
        self._pred_func = pred_func

    def __call__(self, pdf_grid_func, fk_tables, params):
        pdf_params = params[: self.n_pdf_params]
        pdf = pdf_grid_func(pdf_params)
        return self._pred_func(pdf, fk_tables), pdf


def forward_map(_pred_data, pdf_model):
    """Reportengine provider that builds the default FK-table forward map.

    Parameters
    ----------
    _pred_data : callable
        Prediction function of the form ``pred_func(pdf, fk_tables) -> predictions``.
    pdf_model : optional
        Used to infer ``n_pdf_params`` from ``len(pdf_model.param_names)``.

    """

    n_pdf_params = len(pdf_model.param_names)
    return FKTableForwardMap(_pred_data, n_pdf_params=n_pdf_params)
