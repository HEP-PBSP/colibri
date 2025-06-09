.. _how_to_model:

===============================
Implementing a Model in Colibri
===============================

In this section, we will present how to implement a model in Colibri, which can then be used to 
:ref:`run fits <in_running_fits>`, :ref:`closure tests <in_closure_tests>`, and other
applications (see the :ref:`tutorials <in_tutorials>`).

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
The ``pyproject.toml`` file defines a python package for this model. It looks like this:

.. literalinclude:: ../../../../../examples/les_houches_example/pyproject.toml
   :language: python

Note that here the executable ``les_houches_exe`` is introduced, which is an executable
that is specific to this model, and will be used to initialise a fit. 
(See :ref:`Running Fits <in_running_fits>`).

``app.py``
^^^^^^^^^^
The ``app.py`` scripts defines the ``LesHouchesApp`` class, which is based on the abstract
class ``colibriApp``. It looks as follows:

.. literalinclude:: ../../../../../examples/les_houches_example/les_houches_example/app.py
   :language: python

Note that it enables the use of reportengine and validphys functionalities. 
(See :cite:`zahari_kassabov_2019_2571601` for documentation on validphys).

``config.py``
^^^^^^^^^^^^^

The ``config.py`` script defines the configuration layer for the Les Houches 
model. It extends Colibri's configuration system to provide a custom model 
builder and environment.

.. literalinclude:: ../../../../../examples/les_houches_example/les_houches_example/config.py
   :language: python

``model.py``
^^^^^^^^^^^^
the ``model.py`` script defines the Les Houches parametrisation model. It looks like this:

.. literalinclude:: ../../../../../examples/les_houches_example/les_houches_example/model.py
   :language: python

The LesHouchesPDF class is defined, based on the more general class PDFModel, which you can read
more about :ref:`here <pdf_model_class>`. The LesHouchesPDF class does the following:

- takes a list of flavours to be fitted (``param_names``), 
- defines the PDF for each flavour, 
- computes grid values. 

Having defined this model, it can be used to run a fit and perform closure tests. You can find
an example of how to execute the model :ref:`here <lh-closure-test>`.
