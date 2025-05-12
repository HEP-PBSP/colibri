.. _theory_predictions:

===========
Forward Map
===========

Colibri provides a platform for fitting parton distribution function (PDF) models to data which has at least
one incoming proton.
The data is modeled in the framework of collinear QCD factorisation where the scattering process process can be 
written as a convolution of the parton distribution functions (PDFs) with a hard scattering cross section.
For example, the the :math:`F_2` structure function can be decomposed in terms of hard-scattering coefficient 
unctions and PDFs as,

.. math::

    \begin{align} 
    \label{eq:ev} 
    F_2(x,Q^2) &= \sum_i^{n_f} C_i(x,Q^2) \otimes f_i(x,Q^2) \nonumber \\
    &= \sum_{i,j}^{n_f} C_i(x,Q^2) \otimes \Gamma_{ij}(Q^2,Q_0^2) \otimes f_j(x,Q_0^2),
    \end{align}

where :math:`C_i(x,Q^2)` are the process-dependent coefficient functions which
can be computed perturbatively as an expansion in the QCD and QED
couplings;  :math:`\Gamma_{ij}(Q^2,Q_0^2)` is an evolution operator, determined by the
solutions of the DGLAP equations, which evolves the PDF from the initial
parameterization scale :math:`Q_0^2` into the hard-scattering scale :math:`Q^2`,
:math:`f_i(x,Q^2_0)` are the PDFs at the parameterization scale, and
:math:`\otimes` denotes the Mellin convolution.



Theoretical predictions in Colibri make use of the standard factorisation theorem for hard scattering processes.

