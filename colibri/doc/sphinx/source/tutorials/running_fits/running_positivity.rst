.. _general_settings:

================
General Settings
================

In this section, we discuss some settings that can be included in
Colibri runcards that are relevant to any type of fit.

.. _running_positivity:

Imposing Positivity Constraints on a Fit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This section describes how to impose positivity constraints on a
Colibri PDF fit.

You may have noticed these settings on some of the runcards shown
in the previous tutorials:

.. code-block::

    positivity:                            # Positivity datasets, used in the positivity penalty.
        posdatasets:
        - {dataset: NNPDF_POS_2P24GEV_F2U, variant: None, maxlambda: 1e6}

    positivity_penalty_settings:
        positivity_penalty: false
        alpha: 1e-7                           
        lambda_positivity: 0  

To impose positivity constraints on a PDF fit, you can simply set
``positivity_penalty`` to ``true``. You can also add more detasets
under ``posdatasets``.