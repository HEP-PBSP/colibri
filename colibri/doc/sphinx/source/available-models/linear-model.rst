.. _linear-model:

============
Linear Model
============

This model was presented in Ref. :cite:alp:`Costantini:2025wxp`.

**Model Repository:**  https://github.com/HEP-PBSP/wmin-model

Keep in mind...
^^^^^^^^^^^^^^^

This model is especially suitable for :ref:`running bayesian fits <in_running_bayesian>`
but, as well as running fits, it allows the user to generate a POD basis
(see Ref. :cite:alp:`Costantini:2025wxp` for details on what this is).

In order to do either of these, this model requires an extra command after
:ref:`evolution <evolution_script>`, namely:

.. code-block:: bash

    python shift_lhapdf_members.py evolved_directory/postfit/evolved_directory

where ``evolved_directory`` is the fit or POD basis directory that should have
previously been evolved.

