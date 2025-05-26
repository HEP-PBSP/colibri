.. _pdf_model:

==================
Colibri PDF Models
==================


In this section, we present ``colibri``'s core functionality, which enables users to easily implement any PDF parametrisation. We then offer a comprehensive overview of the building blocks and general methodology that support Bayesian (and other) sampling of these parametrisations.

.. _pdf_model_class:

The Colibri PDF model class
--------------------------------

In order to decouple the specifics of the implementation of parton distribution functions (PDFs) parametrisations from their numerical inference, 
we introduce an abstract base class ``PDFModel`` in the ``colibri`` framework, see the code-block below for an overview of the class structure.

At its core, a ``PDFModel`` must provide:

- A list of model parameters, representing the degrees of freedom used to describe the PDF.
- A method that defines how these parameters are mapped to PDF values on a specified grid in momentum fraction ``x``, for each parton flavour.

This abstraction enables users to plug in a variety of model architectures ranging from simple parametric forms to more complex neural network-based approaches, while delegating performance-critical tasks such as convolution with pre‐tabulated perturbative kernels and Bayesian sampling of the parameters to optimized external engines.

In practice, each new PDF parametrisation is implemented as its own ``colibri`` application by subclassing ``PDFModel`` and completing the two required methods, as detailed in the Les Houches tutorial (TODO: reference).

.. literalinclude:: ../../../../../pdf_model.py 
   :language: python
   :pyobject: PDFModel


Parameter Specification
~~~~~~~~~~~~~~~~~~~~~~~

Each concrete PDF model must provide a *parameter list* via a ``param_names`` property. This list of strings defines the parameter names (e.g. normalizations, exponents, polynomial coefficients) in a fixed order.

Grid Evaluation Method
~~~~~~~~~~~~~~~~~~~~~~

The core of the ``PDFModel`` class is the ``grid_values_func`` method, which returns a JAX-compatible function

.. math:: 
    f_{\rm grid}(\boldsymbol{\theta}): \mathbb{R}^{N_{\rm p}} \to \mathbb{R}^{N_{\rm fl} \times N_{\rm x}}

mapping an :math:`N_{\rm p}`-dimensional parameter vector :math:`\boldsymbol{\theta}` into the PDF values 
(note: the function actually returns x*PDF values since this is the object convoluted with the FK-tables) 
for each parton flavour index evaluated on the user-provided :math:`x` grid of length :math:`N_{\rm x}`. 
In practice, for a standard PDF fit, the user only needs to define the expression mapping the :math:`x`-grid to the PDF values. 
The framework then automatically handles the construction of all the resources, such as theory predictions, needed for a PDF fit.

Prediction Construction
~~~~~~~~~~~~~~~~~~~~~~~

To compute physical observables (structure functions, cross sections, etc.), one must convolve the PDFs with perturbative coefficient functions. 
In ``colibri`` this is handled via the ``pred_and_pdf_func`` method, which takes the x-grid and a forward map mapping the PDF to the physical observable, 
and produces a function taking as input the PDF parameters and a tuple of fast-kernel arrays

.. math::
    (\boldsymbol{\theta}, FK) \to (\text{predictions}, f_{\rm grid}(\boldsymbol{\theta}))

that (i) evaluates the PDF on the grid via ``grid_values_func``, and (ii) feeds the resulting :math:`N_{\rm fl} \times N_{\rm x}` 
array into the supplied ``forward_map`` to yield a 1D vector of theory predictions for all data points. 

.. note::
    The prediction function is already implemented, however the user is allowed to override it in its own PDF application 
    if the specific model needs extra features.

Design Rationale and Benefits
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Modularity**: New PDF parametrisations can be added by defining only two methods, without touching the core fitting or convolution engines.
- **Performance**: High-performance array computations and GPU compatibility thanks to the framework being written in JAX.
- **Universality**: All PDF models share the same inference methods as well as data and theory predictions, allowing for more reliable comparison and studies of methodological differences.
