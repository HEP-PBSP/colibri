.. _smeft:

SMEFT forward map
=================

Colibri can fit SMEFT Wilson coefficients together with the PDF parameters of
*any* PDF model. The theory predictions computed from the FK tables are
rescaled by SMEFT K-factors:

.. math::

    T_{\rm SMEFT} = T_{\rm SM}\left(1 + \sum_i c_i L_i + \sum_{i,j} c_i c_j Q_{ij}\right),

where :math:`L_i` (and, optionally, :math:`Q_{ij}`) are the SMEFT responses
normalised to the SM prediction they were computed with. The Wilson
coefficients are appended to the PDF parameters, so they appear in the
priors, initialisation and fit outputs under the names given in the runcard.

Choosing the forward map
------------------------

The forward map is selected with the ``forward_map_type`` key:

- ``fktable`` (default): plain FK-table predictions.
- ``smeft``: FK-table predictions times SMEFT K-factors, configured with
  ``smeft_settings``.

.. code-block:: yaml

    forward_map_type: smeft

    smeft_settings:
      backend: simunet        # source of the K-factors (default: simunet)
      order: EFT_NLO          # default K-factor order for the datasets below
      parameters:
        - {name: OtG, scale: 50.0}
        - name: Y
          linear_combination: {Olq1: 1.51606, Oed: -6.0606, Oeu: 12.1394,
                               Olu: 6.0606, Old: -3.0394, Oqe: 3.0394}
          scale: 200.0
      datasets:
        - {dataset: ATLAS_TTBAR_13TEV_TOT_X-SEC, order: EFT_LO}
        - {dataset: ATLAS_Z0_7TEV_49FB_HIMASS}

- ``parameters``: the SMEFT fit parameters. A parameter is a single operator
  (by default, the one with the same name) or a ``linear_combination`` of
  operators. The fitted value is ``scale * c``, with ``c`` the physical Wilson
  coefficient; ``scale`` defaults to 1.
- ``datasets``: the datasets receiving a SMEFT correction, each with the
  perturbative ``order`` of its K-factors (falling back to the top-level
  ``order``). All other datasets are SM only. Every dataset listed here must
  also appear in ``dataset_inputs``.

Since the fit parameters include the Wilson coefficients, per-parameter prior
bounds or initialiser bounds must list them too, e.g.
``prior_distribution_specs: {bounds: {..., OtG: [-5, 5], Y: [-5, 5]}}``.

K-factor backends
-----------------

Where the K-factors come from is delegated to a backend
(:class:`colibri.smeft.SMEFTBackend`), selected with ``smeft_settings: {backend: ...}``.
Cuts, data ordering, scales and the forward map itself do not depend on the
backend.

- ``simunet``: reads the SIMU-factor files of
  `SIMUnet <https://github.com/HEP-PBSP/SIMUnet>`_. SIMUnet locates its
  ``simudata`` folder from its source tree, so it must be installed from a
  source checkout with ``pip install -e .``.

A new backend only has to return the responses of one dataset, for all its
points (before cuts):

.. code-block:: python

    from colibri.smeft import DatasetResponses, SMEFTBackend, register_smeft_backend

    @register_smeft_backend("smefit")
    class SMEFiTBackend(SMEFTBackend):
        def dataset_responses(self, dataset, order, parameters):
            # linear: (n_params, n_points), quadratic: (n_params, n_params, n_points)
            ...
            return DatasetResponses(linear=linear, quadratic=quadratic)

.. note::

    The forward map is non-linear in the parameters (PDF times Wilson
    coefficients), so the analytic fit cannot be used with ``forward_map_type: smeft``.

.. note::

    The ``forward_map`` resource is now produced by the config
    (``colibriConfig.produce_forward_map``). Models that used to override the
    ``forward_map`` provider should instead override ``produce_forward_map`` in
    their config class.
