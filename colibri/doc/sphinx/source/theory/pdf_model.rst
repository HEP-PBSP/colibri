.. _pdf_model:

==================
Colibri PDF Models
==================

In this section, we present Colibri's core functionality, which enables users to
easily implement any PDF parametrisation. We then offer a comprehensive overview
of the building blocks and general methodology that support Bayesian (and other)
sampling of these parametrisations.

.. _pdf_model_class:

The Colibri PDF model class
--------------------------------

In order to separate the definition of PDF parametrisations from their numerical
inference, we introduce an abstract base class, ``PDFModel``, in the Colibri
framework. See the code-block below for an overview of the class structure.

At its core, a ``PDFModel`` must provide:

- A list of model parameters, representing the degrees of freedom used to describe the PDF.
- A method that defines how these parameters are mapped to PDF values on a specified grid in momentum fraction :math:`x`, for each parton flavour.

This abstraction enables users to plug in a variety of model architectures,
ranging from simple parametric forms to more complex, neural network-based
approaches, while delegating performance-critical tasks to optimized external
engines. These tasks include convolution with pre-tabulated perturbative
kernels and Bayesian sampling of the parameters.

In practice, each new PDF parametrisation is implemented as its own Colibri
application by subclassing ``PDFModel`` and completing the two required methods,
as detailed in the
:ref:`tutorial on building the Les Houches parametrisation model <in_les_houches>`.

.. literalinclude:: ../../../../pdf_model.py 
   :language: python
   :pyobject: PDFModel


Parameter Specification
~~~~~~~~~~~~~~~~~~~~~~~

Each implemented PDF model must provide a *parameter list* via a ``param_names``
property. This list of strings defines the parameter names (e.g. normalisations,
exponents, polynomial coefficients) in a fixed order.

Grid Evaluation Method
~~~~~~~~~~~~~~~~~~~~~~

The core of the ``PDFModel`` class is the ``grid_values_func`` method, which
returns a JAX-compatible function

.. math:: 
    f_{\rm grid}(\boldsymbol{\theta}): \mathbb{R}^{N_{\rm p}} \to \mathbb{R}^{N_{\rm fl} \times N_{\rm x}}

mapping an :math:`N_{\rm p}`-dimensional parameter vector :math:`\boldsymbol{\theta}`
into the PDF values for each parton flavour index evaluated on the user-provided
:math:`x` grid of length :math:`N_{\rm x}`. Note that the function actually returns
:math:`x \times PDF values`, since this is the object convoluted with the FK-tables.

In practice, for a standard PDF fit, the user only needs to define the expression
mapping the :math:`x`-grid to the PDF values. The framework then automatically handles
the construction of all the resources, such as theory predictions, needed for a PDF fit.

Prediction Construction
~~~~~~~~~~~~~~~~~~~~~~~

To compute physical observables (structure functions, cross sections, etc.), PDFs must
be convolved with perturbative coefficient functions. In Colibri, this is handled via
the ``forward_map`` function, which takes the PDF values on the grid and maps them to predictions for the physical observables.

.. math::
    (f_{\rm grid}(\boldsymbol{\theta}), FK) \to \text{predictions}

.. note::
    Although the prediction function is already implemented, the user is allowed to override
    it in their own PDF application if the specific model needs extra features.

Design Rationale and Benefits
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Modularity**: New PDF parametrisations can be added by defining only two methods, without touching the core fitting or convolution engines.
- **Performance**: High-performance array computations and GPU compatibility thanks to the framework being written in JAX.
- **Universality**: All PDF models share the same inference methods as well as data and theory predictions, allowing for more reliable comparison and studies of methodological differences.
