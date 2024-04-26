.. _installation:

Installing Colibri
==================


Installing colibri and the models together:
--------------------------------------------

.. code-block:: yaml

    cd colibri
    conda env create -f full_environment.yml


Installing colibri and the models separately:
----------------------------------------------

For colibri:

.. code-block:: yaml

    cd colibri
    conda env create -f environment.yml

For the models:

.. code-block:: yaml

    cd colibri/models/MODEL_NAME
    pip install -e .
    
