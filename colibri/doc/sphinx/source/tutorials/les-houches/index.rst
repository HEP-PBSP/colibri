.. _les_houches:

===========================
Les Houches Parametrisation
===========================

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

After having established what model we will be fitting, we will show how the 
:ref:`implementation <lh_implementation>` of the model in Colibri is done.

Finally, we will use the model to perform a model-specific closure test, and
a Closure Test with a PDF set. TODO: add references here to the relevant sections.




.. toctree::
   :maxdepth: 1
   
   ./lh-theory
   ./lh-implementation 
