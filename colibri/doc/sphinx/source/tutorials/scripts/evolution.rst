.. _evolution_script:

Evolution script
----------------

The Colibri evolution script is a wrapper around the `evolven3fit` script
that only allows for the ``evolve`` option. (See the
:mod:`colibri.scripts.evolve_fit` module and the
:func:`colibri.scripts.evolve_fit.main` function).


It can be executed from the command line as follows:

.. code-block:: bash

   evolve_fit <name_fit>

where ``<name_fit>`` is the name of the fit you want to evolve. The script
also has a ``--help`` option that will show you all the options available.
If the performed fit is a fit which requires ``ErrorType=hessian``, 
i.e. the PDF grids to evolve are eigenvectors of the covariance matrix, 
you can specify the option

.. code-block:: bash

   evolve_fit <name_fit> --hessian_fit True

to correctly produce the LHAPDF set.
For more information on the evolution see also the helper from the
``evolven3fit`` script.

Postfit emulation
^^^^^^^^^^^^^^^^^

For Bayesian fits, we don't do any postfit selection on the posterior.
However, for backwards compatibility with the `validphys` module, we
still run a postfit emulation which takes care of creating the central
replica and a ``postfit`` folder containing the evolved replicas as well
as the corresponding LHAPDF set.

Upload of the fit
^^^^^^^^^^^^^^^^^

After running the evolution script, it is possible (if you have the right permissions)
to upload the fit to the `validphys` server using the `validphys` script:

.. code-block:: bash

   vp-upload <name_fit>

After which the fit can be installed and made available in the environment with the
following command:

.. code-block:: bash

   vp-get fit <name_fit>

If you don't have the right permissions, we recommend that you symlink the LHAPDF set to
the LHAPDF environment folder or to symlink the fit folder to the ``NNPDF/results`` folder
of your environment.

.. note::

    The final folder after the evolution will also contain a symlink `nnfit -> replicas`
    needed for `validphys` and `evolven3fit`, as well as a ``postfit`` folder.