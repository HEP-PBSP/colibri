.. _in_closure_tests:

=====================
Running Closure Tests
=====================

In this section, we demonstrate how to closure tests using Colibri.
In the closure test tutorial, we run a standard closure test using the
Les Houches parametrisation as a model. :ref:`This tutorial <in_les_houches>`
covers how to implement this model in Colibri.

In the model-specific closure test tutorial, we cover the settings that need to
be added to the runcard to run a model-specific closure test.

.. note::

   In NNPDF terminology, a closure test is a predictive-posterior check.  
   For further details, see :cite:`Barontini:2025lnl` and :cite:`DelDebbio:2021whr`.

If you haven't already done so, you will need to follow the 
:ref:`installation instructions <installation>` first.

.. toctree::
   :maxdepth: 1
   
   ./lh-closure-test
   ./model-cl-test