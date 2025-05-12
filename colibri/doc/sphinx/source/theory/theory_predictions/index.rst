.. _theory_predictions:

===========
Forward Map
===========

Colibri provides a flexible platform for fitting parton distribution function (PDF) models to data involving at least
one incoming proton. The data are modeled in the framework of collinear QCD factorization, where the scattering
process is written as a convolution of the PDFs with perturbatively-computed hard-scattering cross sections.

In this context, inferring the PDFs from experimental measurements is an inverse problem: the unknowns are the
PDFs, and the forward model consists of the hard-scattering cross section combined with PDF evolution kernels,
commonly stored as FK tables. For more details on FK tables and evolution kernels, see :cite:`Candido:2022tld`.

We distinguish two classes of forward maps based on whether the initial state involves one proton (Deep Inelastic
Scattering, DIS) or two protons (hadron–hadron collisions).


DIS predictions
^^^^^^^^^^^^^^^

DIS data is the most abundant data type in the global PDF fits and is the most 
straightforward to model.
For example, a measurement of the :math:`F_2` structure function consisting of :math:`N_{\rm data}` points, 
can be written as the contraction of two operators,

.. math::

    \begin{align}
    \label{eq:dis_prediction}
    F_{2,i} &= \sum_j^{N_{\rm fl}} \sum_{k}^{N_{\rm x}} FK_{i,j,k} \, f_{j,k}  \; ,
    \end{align}

where the :math:`FK_{i,j,k}` operator has shape :math:`(N_{\rm data}, N_{\rm fl}, N_{\rm x})` and :math:`f_{j,k}` 
is the :math:`(N_{\rm fl}, N_{\rm x})` dimensional grid representing the PDF values at the input scale :math:`Q_0^2`.

In colibri, the forward modeling of DIS data is taken care of in the :py:class:`colibri.theory_predictions` module
by the :py:func:`colibri.theory_predictions.make_dis_prediction` closure function which essentially takes in input an 
FK-table object and returns a function that computes the DIS prediction for a given PDF and FK-array.


.. literalinclude:: ../../../../../theory_predictions.py 
   :language: python
   :pyobject: make_dis_prediction



Hadron-Hadron predictions
^^^^^^^^^^^^^^^^^^^^^^^^^

Hadron-hadron collisions are more complicated to model than DIS data, as they involve the convolution of two
incoming partons, each with their own PDF. 
A :math:`N_{\rm data}`-point measurement :math:`\sigma` of a hadron-hadron cross section can be written as

.. math::

    \begin{align}
    \label{eq:had_prediction}
    \sigma_{i} &= \sum_{j,k}^{N_{\rm fl}} \sum_{l,m}^{N_{\rm x}} FK_{i,j,k,l,m} \, f_{j,l} f_{k,m}  \; ,
    \end{align}

where the :math:`FK_{i,j,k,l,m}` operator has shape :math:`(N_{\rm data}, N_{\rm fl}, N_{\rm fl}, N_{\rm x}, N_{\rm x})`.


In colibri, the forward modeling of hadron-hadron data is taken care of in the :py:class:`colibri.theory_predictions` module
by the :py:func:`colibri.theory_predictions.make_had_prediction` closure function shown below.

.. literalinclude:: ../../../../../theory_predictions.py 
   :language: python
   :pyobject: make_had_prediction