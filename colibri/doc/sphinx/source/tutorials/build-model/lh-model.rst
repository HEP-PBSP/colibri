.. _how_to_model:

===============================
Implementing a Model in Colibri
===============================

In this section, we will present how to implement a model in Colibri, which can then be 
used to :ref:`run fits <in_running_fits>`, :ref:`closure tests <in_closure_tests>`, 
and other applications (see the :ref:`tutorials <in_tutorials>`).

In general, a Colibri model is contained in a directory with the following structure:

.. code-block:: text

   model_to_implement/
   ├── pyproject.toml        # Defines a python package for the project and sets up executable
   ├── model_to_implement/   
      ├── app.py             # Enables the use of reportengine and validphys 
      ├── config.py          # Defines the configuration layer for the model
      ├── model.py           # Script where the model is defined
   └── runcards/             # Directory containing any runcards

The best way to understand how to implement a model is to go through an example, so let's have a 
look at how the Les Houches parametrisation (presented :ref:`here <lh_theory>`) is built. 

.. _lh_model:

Implementing the Les Houches model in Colibri
---------------------------------------------

In the ``colibri/examples/`` directory, you will find a directory called ``les_houches_example``,
which follows the sutructure defined above. We will have a look at them one by one. 

``pyproject.toml``
^^^^^^^^^^^^^^^^^^
The ``pyproject.toml`` file defines the Python package configuration for this model using 
`Poetry <https://python-poetry.org/docs/pyproject/>`_ as the dependency management and 
packaging tool. The configuration file structure looks like this:

.. literalinclude:: ../../../../../examples/les_houches_example/pyproject.toml
   :language: python

Note that here the executable ``les_houches_exe`` is introduced, which is an executable
that is specific to this model, and will be used to initialise a fit. 
(See :ref:`Running Fits <in_running_fits>`).

``app.py``
----------
The ``app.py`` module defines the core application class for the Les Houches model:

.. literalinclude:: ../../../../../examples/les_houches_example/les_houches_example/app.py
   :language: python

The ``LesHouchesApp`` class enables the Les Houches model to function as a 
`reportengine App <https://github.com/NNPDF/reportengine>`_. 
This integration provides a structured framework for data processing and report generation.

Key Features:
~~~~~~~~~~~~~

* **Provider System**: The ``LesHouchesApp`` accepts a list of providers (``lh_pdf_providers``) containing modules that are recognized by the application framework.

* **Inheritance Hierarchy**: The ``LesHouchesApp`` is a subclass of ``colibriApp``, which means it automatically inherits all providers from both `colibri` and `validphys`, giving access to their full functionality without additional configuration.


``config.py``
^^^^^^^^^^^^^

The ``config.py`` script defines the configuration layer for the Les Houches 
model. It extends Colibri's configuration system to provide a custom model 
builder and environment.

.. literalinclude:: ../../../../../examples/les_houches_example/les_houches_example/config.py
   :language: python

The ``produce_pdf_model`` method creates an instance of the ``LesHouchesPDF`` model. 
Therefore, every model should have this production rule.

If ``dump_model`` is set to ``True``, the method serialises the model using ``dill`` 
and writes it to ``pdf_model.pkl`` in the ``output_path``, where ``output_path`` will 
be the output directory created when running a colibri fit 
(see :ref:`colibri_fit_folders`). ``pdf_model.pkl`` will be loaded by 
``scripts/bayesian_resampler.py`` for resampling. 

If ``dump_model`` is set to ``False``, the serialised model will not be written to the
disk. 

``model.py``
^^^^^^^^^^^^
The ``model.py`` script defines the Les Houches parametrisation model. It looks like this:

.. literalinclude:: ../../../../../examples/les_houches_example/les_houches_example/model.py
   :language: python

The LesHouchesPDF class completes the abstract methods of the PDFModel class, which you
can read more about :ref:`here <pdf_model_class>`. This allows for the definition of a
specific model in a way that can be used in the colibri code. The LesHouchesPDF class 
does the following:

- takes a list of flavours to be fitted (``param_names``), 
- defines the PDF for each flavour, 
- computes grid values. 

Having defined this model, it is used in the production rule ``produce_pdf_model``,
defined in the ``config.py`` script, shown above. This allows the model to be seen
by the rest of the code, so that it can be used to run a fit and perform closure tests.

You can find an example of how to execute the model :ref:`here <lh-closure-test>`.
