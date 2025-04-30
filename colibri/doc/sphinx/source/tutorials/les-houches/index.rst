.. _les_houches:

===========================
Les Houches Parametrisation
===========================
If you have followed the :ref:`installation instructions <installation>`,
you can follow this tutorial to use Colibri to implement the Les Houches 
parametrisation model, which is described in the :ref:`theory section <lh_theory>`.

In this tutorial, we will show how to use Colibri to perform a fit using the 
Les Houches parametrisation (:cite:`Alekhin:2005xgg`) as our model. 
This parametrisation is simple enough for us to exemplify the use of Colibri, 
while still being realistic enough that this tutorial can be used as a 
template for other, more complex parametrisations or models.

First, we will provide a :ref:`description <lh_theory>` of the Les Houches model,
establishing how many and which are the free parameters to be fitted. since
Colibri works in the `evolution basis`, we will also 
:ref:`describe in detail <lh-evolution-basis>` what the PDFs to be fitted look 
like in this basis. 

After having established what model we will be fitting, we will show how a model
is built in Colibri, by :ref:`implementing the Les Houches model <lh_model>`.

Finally, we will use the model to perform a :ref:`model-specific closure test <lh-model-specific-closure-test>`, 
and a :ref:`closure test with a PDF set <lh-closure-test>`. 




.. toctree::
   :maxdepth: 1
   
   ./lh-theory
   ./lh-model
   ./lh-closure-test
   ./lh-model-specific-closure-test
