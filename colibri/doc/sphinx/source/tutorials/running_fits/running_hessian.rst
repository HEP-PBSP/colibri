.. _running_hessian:

============
Hessian Fits
============

This section describes how to run Hessian fits with Colibri.

Hessian fits are performed by minimising the :math:`\chi^2` loss defined with respect
to the experimental data. The best-fit parameters are found by minimising the :math:`\chi^2`
using gradient descent. The Hessian matrix is then computed at the minimum and used to define
the covariance matrix of the fit parameters. A parameter called tolerance can be used to inflate
the uncertainties, as is common in global PDF fits. For more details on the Hessian method, see
Refs. :cite:alp:`Bailey:2020ooq` and :cite:alp:`Pumplin:2002vw`.

In this tutorial, we will present an example that can be used to fit the
:ref:`Les Houches parametrisation model <lh_theory>`.

Runcard
-------

The runcard ``runcards/lh_hessian.yaml`` in the ``examples/les_houches_example``
directory runs a Hessian fit of the Les Houches model.

.. literalinclude:: ../../../../../examples/les_houches_example/runcards/lh_hessian.yaml
    :language: yaml


Note that the Hessian fit uses the same ``param_initialiser_settings`` as a
Monte Carlo fit, and so any of the initialisation options discussed in the
:ref:`Monte Carlo fit tutorial <param-initialiser-settings>` can be used for these
settings (i.e. gaussian initialisation, global bounds for all parameters ...).

``hessian_settings``
^^^^^^^^^^^^^^^^^^^^
* ``tolerance``: 
    This parameter inflates the uncertainties obtained from the Hessian matrix by
    multiplying the covariance matrix by :math:`T^2`.
* ``n_eigvec``: 
    Number of eigenvectors to use. If not set, defaults to 20.
* ``iter_init``:
    Set this to however many different starting points to try to initialise from.
    The one with the lowest :math:`\chi^2` will be taken. This is done in order to avoid local minima.
* ``rng_seed``:
    Controls the random number generator seed for reproducibility.
* ``grad_tol``:
    Gradient-norm tolerance used in the local-minimum check. If the norm of the 
    gradient at the candidate minimum is smaller than this value, the point is considered 
    approximately stationary. Default is :math:`1\times 10^{-6}`.
* ``min_hessian_eigval``:
    Minimum allowed eigenvalue of the Hessian for the positive-definiteness check. 
    If the smallest eigenvalue is below this threshold, the Hessian is considered 
    not positive definite. Default is :math:`1\times 10^{-12}`.
* ``require_local_min``:
    Whether to enforce convergence to a verified local minimum. If ``True`` and 
    the local-minimum checks (gradient small + Hessian positive definite) fail, 
    the fit will raise an error. If ``False``, the fit will continue with a warning.

Running the fit
---------------

To run a Hessian fit for the Les Houches model 
(presented in :ref:`this tutorial <in_les_houches>`), you would run (in case you have not done so already):

.. code-block:: bash

    pip install -e .

from the ``examples/les_houches_example`` directory. This enables the
model-specific executable.

Then, from the same directory, you can run the above runcard with the following command:

.. code-block:: bash

    les_houches_exe runcards/lh_hessian.yaml

Finally, you can evolve the fit, as described in :ref:`this section <evolution_script>`.
For Hessian fits, evolution needs to be performed with the flag ``--hessian_fit True``.