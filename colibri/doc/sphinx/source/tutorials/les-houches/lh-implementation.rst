.. _lh_implementation:

==============
Implementation
==============

If you have followed the :ref:`installation instructions <installation>`,
you can follow this tutorial to use Colibri to implement the Les Houches 
parametrisation model, which is described in the :ref:`theory section <lh_theory>`.

First, we will have a look at :ref:`how the model is implemented <lh-model-implementation>`, 
and then, we will use it to do a model-specific and closure test and a closure test 
with a PDF grid. TODO: add relevant references to sections.

.. _lh-model-implementation:

Building the Les Houches model in Colibri
-----------------------------------------

To understand how a model is implemented in Colibri, let's have a look at how the Les Houches
example model. In the ``colibri/examples/les_houches_example/les_houches_example`` directory,
you will find a script called ``model.py``. It looks like this: 




.. _lh-model-specific-closure-test:

Model-Specific Closure Test 
---------------------------

The first step is to download the package for this tutorial. 

In your ``colibri-dev`` conda environment, go to the ``les_houches_example`` directory, found in:

.. code-block:: bash
    
    cd colibri/examples/les_houches_example 

Then run:

.. code-block:: bash
    
    pip install -e .

This will enable an exexutable called ``les_houches_exe``. 

In the directory ``les_houches_example/les_houches_example`` you will find a file named ``model.py``, in which the Les Houches parametrisation is implemented, follwing the
description in the :ref:`theory section <lh_theory>`. 

In order to run the example, go to the directory ``les_houches_example/runcards``, where you will find the ``les_houches_runcard.yaml`` runcard. Then run:

.. code-block:: bash
    
    les_houches_exe les_houches_runcard.yaml -o les_houches_output

This will run colibri and create a directory called ``les_houches_output``, where you will find the output of the run.