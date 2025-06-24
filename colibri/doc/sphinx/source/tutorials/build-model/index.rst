.. _in_les_houches:

========================
Implementing a PDF model
========================

If you have followed the :ref:`installation instructions <installation>`,
you can follow this tutorial to use Colibri to implement the Les Houches 
parametrisation model, which is described in the 
:ref:`theory section <lh_theory>`.

This parametrisation is simple enough for us to exemplify the use of Colibri, 
while still being realistic enough that this tutorial can be used as a 
template for other, more complex parametrisations or models.

First, we will provide a :ref:`description <lh_theory>` of the Les Houches model,
establishing how many and which are the free parameters to be fitted. since
Colibri works in the `evolution basis`, we will also 
:ref:`describe in detail <lh-evolution-basis>` what the PDFs to be fitted look 
like in this basis. 

After establishing how the Les Houches model looks like, we will 
:ref:`implement it in Colibri <lh_model>`.

This model can then be used to perform fits and run closure tests (see
:ref:`tutorials <in_tutorials>`).


.. toctree::
   :maxdepth: 1
   
   ./lh-theory
   ./lh-model
