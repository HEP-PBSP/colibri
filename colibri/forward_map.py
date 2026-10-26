"""
colibri.forward_map.py

Forward maps: parameters → theory predictions.

A ``ForwardMap`` implements the final stage of the fit pipeline, turning the
fit parameter vector into theory predictions that can be compared with
data in the likelihood. It will also return the PDF values on the fit x-grid,
which is sometimes needed for computing penalties.


Design choice: fixed call signature
-----------------------------------
The log-likelihood calls every forward map with the same fixed signature::

    (fk_tables, params) -> predictions, pdf

The PDF grid function is bound at construction time (via ``pdf_grid_func``
stored on the instance), so ``fk_tables`` remain a dynamic argument that JAX
traces as an abstract input — keeping them out of the compiled binary and
avoiding expensive recompilation when values change.

Parameter convention
--------------------
``params`` is a 1-D array containing *all* fit parameters. In colibri we allow
for "extra" fit parameters beyond the PDF model parameters (e.g. nuisance-like factors,
or parameters of a custom prediction function).

By convention:

``params[:self.n_pdf_params]`` are PDF parameters consumed by the bound
``pdf_grid_func``; any remaining entries are "extra" parameters interpreted by
the chosen ``ForwardMap`` implementation.

Choosing the forward map
------------------------
The ``forward_map`` resource is produced by ``colibriConfig.produce_forward_map``
according to the ``forward_map_type`` runcard key:

* ``fktable`` (default): :func:`fktable_forward_map`, plain FK-table convolution.
* ``smeft``: :func:`smeft_forward_map`, FK-table convolution times SMEFT
  K-factors, configured with ``smeft_settings`` (see :mod:`colibri.smeft`).
  With a PDF model without parameters (``fixed_pdf``) the SM predictions are
  computed once and only the Wilson coefficients are fitted.

Example - fitting a normalisation factor on top of the PDF
----------------------------------------------------------
::

    class NormForwardMap(ForwardMap):
        def __init__(self, pred_func, pdf_model, pdf_grid_func):
            super().__init__(pdf_model, extra_param_names=["norm"])
            self._pred_func = pred_func
            self._pdf_grid_func = pdf_grid_func

        def __call__(self, fk_tables, params):
            pdf = self._pdf_grid_func(params[: self.n_pdf_params])
            norm = params[self.n_pdf_params]            # first extra parameter
            return norm * self._pred_func(pdf, fk_tables), pdf

Example - fixed PDF, fitting only extra parameters
--------------------------------------------------
::

    class FixedPDFForwardMap(ForwardMap):
        def __init__(self, pred_func, fixed_pdf, fk_tables, pdf_model=None):
            super().__init__(pdf_model)
            self._pred_func = pred_func
            self.fixed_pdf = fixed_pdf
            self._fixed_pred = self._pred_func(fixed_pdf, fk_tables)

        def __call__(self, fk_tables, params):
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

        ``(fk_tables, params) -> predictions``

    Notes
    -----
    The split point between PDF parameters and "extra" parameters is owned
    by the forward map via ``self.n_pdf_params``.
    """

    def __init__(self, pdf_model, extra_param_names: list[str] = []):

        self.pdf_model = pdf_model
        if pdf_model is not None:
            self.pdf_param_names = pdf_model.param_names
        else:
            self.pdf_param_names = []
        self.extra_param_names = extra_param_names

    @property
    def n_pdf_params(self) -> int:
        """Number of PDF parameters, derived from ``pdf_param_names``."""
        return len(self.pdf_param_names)

    @property
    def param_names(self) -> list[str]:
        """All fit parameter names: PDF parameters followed by extra parameters."""
        return list(self.pdf_param_names) + list(self.extra_param_names)

    @abstractmethod
    def __call__(
        self,
        fk_tables: Any,
        params: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute theory predictions from fit parameters.

        Parameters
        ----------
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
        self,
        pred_func: Callable[[jnp.ndarray, Any], jnp.ndarray],
        pdf_model,
        pdf_grid_func: Callable[[jnp.ndarray], jnp.ndarray],
        extra_param_names: list[str] = [],
    ):
        super().__init__(pdf_model, extra_param_names=extra_param_names)
        self._pred_func = pred_func
        self._pdf_grid_func = pdf_grid_func

    def __call__(self, fk_tables, params):
        pdf_params = params[: self.n_pdf_params]
        pdf = self._pdf_grid_func(pdf_params)
        return self._pred_func(pdf, fk_tables), pdf


def fktable_forward_map(_pred_data, pdf_model, FIT_XGRID, extra_param_names=[]):
    """Reportengine provider that builds the default FK-table forward map.

    Selected with ``forward_map_type: fktable`` (the default), see
    ``colibriConfig.produce_forward_map``.

    Parameters
    ----------
    _pred_data : callable
        Prediction function of the form ``pred_func(pdf, fk_tables) -> predictions``.
    pdf_model : object
        The PDF model object; must expose ``param_names`` and ``grid_values_func``.
    FIT_XGRID : array-like
        The x-grid on which the PDF is evaluated.
    extra_param_names : list[str], optional
        Names of any additional fit parameters beyond the PDF parameters.

    """
    pdf_grid_func = pdf_model.grid_values_func(FIT_XGRID)
    return FKTableForwardMap(
        pred_func=_pred_data,
        pdf_model=pdf_model,
        pdf_grid_func=pdf_grid_func,
        extra_param_names=extra_param_names,
    )


class SMEFTForwardMap(FKTableForwardMap):
    """FK-table forward map with SMEFT K-factors on top of the SM predictions::

        T = T_SM(pdf) * (1 + c @ L + c @ Q @ c)

    The extra fit parameters are the SMEFT parameters ``w = scale * c``, with
    ``c`` the physical Wilson coefficients and ``L`` (``Q``) the linear
    (quadratic) responses held by :class:`colibri.smeft.SMEFTCorrections`.

    If the PDF model has no parameters (e.g. ``fixed_pdf``), ``T_SM`` does not
    depend on the fit parameters: it is computed once here from ``fk_tables``,
    and the FK tables passed at call time are ignored. Without quadratic
    responses the predictions are then linear in the Wilson coefficients, so
    the analytic fit can be used.
    """

    def __init__(
        self,
        pred_func: Callable[[jnp.ndarray, Any], jnp.ndarray],
        pdf_model,
        pdf_grid_func: Callable[[jnp.ndarray], jnp.ndarray],
        smeft_corrections,
        fk_tables: Any = None,
    ):
        super().__init__(
            pred_func,
            pdf_model,
            pdf_grid_func,
            extra_param_names=list(smeft_corrections.parameter_names),
        )
        self.scales = jnp.asarray(smeft_corrections.scales)
        self._linear = jnp.asarray(smeft_corrections.linear)
        self._quadratic = (
            None
            if smeft_corrections.quadratic is None
            else jnp.asarray(smeft_corrections.quadratic)
        )

        # SM predictions and PDF grid, precomputed when the PDF is fixed
        self.sm_predictions = None
        self.pdf_grid = None
        if self.n_pdf_params == 0:
            if fk_tables is None:
                raise ValueError(
                    "SMEFTForwardMap needs fk_tables when the PDF model has no "
                    "parameters, to compute the SM predictions once."
                )
            self.pdf_grid = jnp.asarray(pdf_grid_func(jnp.array([])))
            self.sm_predictions = jnp.asarray(pred_func(self.pdf_grid, fk_tables))

    def physical_coefficients(self, params):
        """Physical Wilson coefficients ``c`` from the full parameter vector."""
        return params[self.n_pdf_params :] / self.scales

    def smeft_kfactor(self, params):
        """K-factor multiplying the SM predictions, one entry per data point."""
        c = self.physical_coefficients(params)
        kfactor = 1.0 + c @ self._linear
        if self._quadratic is not None:
            kfactor = kfactor + jnp.einsum("i,j,ijn->n", c, c, self._quadratic)
        return kfactor

    def __call__(self, fk_tables, params):
        if self.sm_predictions is not None:
            sm_predictions, pdf = self.sm_predictions, self.pdf_grid
        else:
            sm_predictions, pdf = super().__call__(fk_tables, params)
        return sm_predictions * self.smeft_kfactor(params), pdf


def smeft_forward_map(
    _pred_data, pdf_model, FIT_XGRID, smeft_corrections, fast_kernel_arrays
):
    """Reportengine provider that builds the SMEFT forward map.

    Selected with ``forward_map_type: smeft``, see
    ``colibriConfig.produce_forward_map``. The K-factors are configured with
    the ``smeft_settings`` runcard key.

    Parameters
    ----------
    _pred_data : callable
        Prediction function of the form ``pred_func(pdf, fk_tables) -> predictions``.
    pdf_model : object
        The PDF model object; must expose ``param_names`` and ``grid_values_func``.
    FIT_XGRID : array-like
        The x-grid on which the PDF is evaluated.
    smeft_corrections : colibri.smeft.SMEFTCorrections
        SMEFT responses for the fitted data.
    fast_kernel_arrays : tuple
        FK arrays, only used to precompute the SM predictions when the PDF
        model has no parameters (e.g. ``fixed_pdf``).
    """
    return SMEFTForwardMap(
        pred_func=_pred_data,
        pdf_model=pdf_model,
        pdf_grid_func=pdf_model.grid_values_func(FIT_XGRID),
        smeft_corrections=smeft_corrections,
        fk_tables=fast_kernel_arrays,
    )
