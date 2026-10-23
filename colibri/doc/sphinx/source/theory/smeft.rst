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

Template runcards are in ``colibri/examples/les_houches_example/runcards``:
``lh_smeft_fit_closure_test.yaml`` (Les Houches PDF and Wilson coefficients
fitted together) and ``smeft_fixed_pdf_fit.yaml`` (SMEFT-only fit with a fixed
PDF).

The forward map is selected with the ``forward_map_type`` key:

- ``fktable`` (default): plain FK-table predictions.
- ``smeft``: FK-table predictions times SMEFT K-factors, configured with
  ``smeft_settings`` and the ``smeft_order`` key of ``dataset_inputs``.

.. code-block:: yaml

    dataset_inputs:
    - {dataset: SLAC_NC_NOTFIXED_P_EM-F2, variant: legacy_dw}    # SM only
    - {dataset: ATLAS_TTBAR_13TEV_TOT_X-SEC, variant: legacy, smeft_order: EFT_NLO}
    - {dataset: ATLAS_Z0_7TEV_49FB_HIMASS, variant: legacy, smeft_order: EFT_LO}

    forward_map_type: smeft

    smeft_settings:
      backend: simunet        # source of the K-factors (default: simunet)
      # backend_options: {simudata_path: /path/to/SIMUnet/simudata}  # only if needed, see below
      parameters:
        - {name: OtG, scale: 50.0}
        - name: Y
          linear_combination: {Olq1: 1.51606, Oed: -6.0606, Oeu: 12.1394,
                               Olu: 6.0606, Old: -3.0394, Oqe: 3.0394}
          scale: 200.0

- ``smeft_order`` (in ``dataset_inputs``): marks the datasets receiving a
  SMEFT correction, and the perturbative order of their K-factors, as named
  by the backend (``EFT_LO`` or ``EFT_NLO`` for SIMUnet). Datasets without it
  are SM only. It is ignored, with a warning, unless ``forward_map_type: smeft``.
- ``parameters``: the SMEFT fit parameters. A parameter is a single operator
  (by default, the one with the same name) or a ``linear_combination`` of
  operators. The fitted value is ``scale * c``, with ``c`` the physical Wilson
  coefficient; ``scale`` defaults to 1.

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
  `SIMUnet <https://github.com/scarlehoff/SIMUnet>`_. Install it with the
  ``smeft`` extra::

      pip install "colibri[smeft]"

  The K-factor files are not part of the SIMUnet Python package: they live in
  the ``simudata`` folder at the top of the SIMUnet repository, and SIMUnet
  looks for them two levels above its installed package. Hence:

  - SIMUnet installed from a clone with ``pip install -e .``: the package is
    ``<clone>/src/simunet``, so SIMUnet finds ``<clone>/simudata`` by itself.
    Nothing to set.
  - SIMUnet installed with ``pip install "colibri[smeft]"`` (or any
    non-editable install): the package is copied to ``site-packages`` without
    the K-factor files. Clone the repository and point to its ``simudata``
    folder:

    .. code-block:: yaml

        smeft_settings:
          backend: simunet
          backend_options:
            simudata_path: /path/to/SIMUnet/simudata

  If the files cannot be found, the fit stops at the start with an error
  saying so.

``backend_options`` are passed to the backend constructor, so each backend can
take its own options.

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

    When fitting the PDF too, the forward map is non-linear in the parameters
    (PDF times Wilson coefficients), so the analytic fit cannot be used. With a
    fixed PDF and linear K-factors it can, see below.

SMEFT-only fits with a fixed PDF
--------------------------------

To fit only the Wilson coefficients, fix the PDF to any LHAPDF set with the
``fixed_pdf`` key and run the plain ``colibri`` executable:

.. code-block:: yaml

    fixed_pdf: NNPDF40_nnlo_as_01180
    fixed_pdf_member: 0          # default; an integer or "replica"

    forward_map_type: smeft
    smeft_settings: {...}

    prior_settings:
      prior_distribution: uniform_parameter_prior
      prior_distribution_specs: {min_val: -5, max_val: 5}

    actions_:
    - run_analytic_fit           # or run_ultranest_fit, run_monte_carlo_fit, ...

- The PDF model then has no parameters: the fit parameters are only the
  Wilson coefficients, and the SM predictions are computed once at the start
  instead of at every likelihood evaluation.
- With linear K-factors (all SIMUnet ones) the predictions are linear in the
  coefficients, so ``run_analytic_fit`` gives the exact Gaussian posterior.
- ``fixed_pdf_member`` chooses the member of the set (0 is the central value).
  With ``fixed_pdf_member: replica``, a Monte Carlo fit run with ``-rep N``
  uses replica ``N`` of the set (which must be a Monte Carlo set), so that the
  PDF uncertainty is propagated to the coefficients across replicas.
- No PDF replicas or exportgrids are written, and there is nothing to run
  ``evolve_fit`` on. ``mc_postfit`` and the resampling scripts still work on
  the coefficients.
- Positivity and integrability penalties are constant with a fixed PDF; a
  warning is printed if they are switched on.
- ``fixed_pdf`` is read by the base colibri config; PDF model apps define their
  own ``produce_pdf_model`` and ignore it.

.. note::

    The ``forward_map`` resource is now produced by the config
    (``colibriConfig.produce_forward_map``). Models that used to override the
    ``forward_map`` provider should instead override ``produce_forward_map`` in
    their config class.
