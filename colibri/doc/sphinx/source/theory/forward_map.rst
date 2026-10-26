.. _forward_map:

Forward map
===========

The forward map turns the fit parameters into the theory predictions that the
likelihood compares with the data. It is the ``forward_map`` resource, an
instance of :class:`colibri.forward_map.ForwardMap`, called as

.. code-block:: python

    predictions, pdf = forward_map(fk_tables, params)

- ``fk_tables``: the FK arrays of the fitted datasets. They are passed at call
  time, rather than stored in the forward map, so that JAX treats them as an
  input and does not compile them into the likelihood.
- ``params``: 1-D array with *all* the fit parameters.
- ``predictions``: 1-D array with one prediction per data point.
- ``pdf``: the PDF grid on the fit :math:`x`-grid, shape
  :math:`(N_{\rm fl}, N_{\rm x})`, used by the positivity and integrability
  penalties.

Fit parameters
--------------

The fit parameters are the parameters of the PDF model, followed by any
*extra* parameters of the forward map:

- ``forward_map.param_names``: all the parameter names, in this order. Priors,
  initialisation and fit outputs use these names.
- ``params[:forward_map.n_pdf_params]``: the PDF parameters.
- ``params[forward_map.n_pdf_params:]``: the extra parameters, e.g. the SMEFT
  Wilson coefficients.

Available forward maps
----------------------

The forward map is selected with the ``forward_map_type`` runcard key:

- ``fktable`` (default): the PDF grid is convolved with the FK tables (see
  :ref:`theory_predictions`). The fit parameters are the PDF parameters only.
- ``smeft``: the FK-table predictions times SMEFT K-factors, with the Wilson
  coefficients as extra parameters. See :ref:`smeft`.

Custom forward maps
-------------------

The choice is made by ``colibriConfig.produce_forward_map``. A model that needs
a different forward map subclasses :class:`~colibri.forward_map.ForwardMap`,
writes a provider that builds it, and overrides ``produce_forward_map`` in its
config class. For example, a forward map fitting an overall normalisation on
top of the PDF:

.. code-block:: python

    # my_model/forward_map.py (a provider module of the app)
    from colibri.forward_map import ForwardMap

    class NormForwardMap(ForwardMap):
        def __init__(self, pred_func, pdf_model, pdf_grid_func):
            super().__init__(pdf_model, extra_param_names=["norm"])
            self._pred_func = pred_func
            self._pdf_grid_func = pdf_grid_func

        def __call__(self, fk_tables, params):
            pdf = self._pdf_grid_func(params[: self.n_pdf_params])
            norm = params[self.n_pdf_params]
            return norm * self._pred_func(pdf, fk_tables), pdf

    def norm_forward_map(_pred_data, pdf_model, FIT_XGRID):
        return NormForwardMap(
            _pred_data, pdf_model, pdf_model.grid_values_func(FIT_XGRID)
        )

.. code-block:: python

    # my_model/config.py
    from reportengine.configparser import explicit_node

    from colibri.config import colibriConfig
    from my_model.forward_map import norm_forward_map

    class MyConfig(colibriConfig):
        @explicit_node
        def produce_forward_map(self, forward_map_type="fktable", dataset_inputs=None):
            if forward_map_type == "norm":
                return norm_forward_map
            # Keep the built-in forward maps available
            return super().produce_forward_map(forward_map_type, dataset_inputs).value

The runcard then selects it with ``forward_map_type: norm``, and ``norm``
appears among the fit parameters.
