.. _closure_tests:

==========================
Closure Testing in Colibri
==========================

In this section we describe what we mean by a "closure test" in Colibri, and
what the differences are between different levels of closure tests.

For a tutorial on how to run closure tests see :ref:`this section <in_closure_tests>`.

What is a Closure Test?
-----------------------

Closure testing is a method to assess the validity of the fitting methodology.

It tries to answer the question:

*Can my fitting methodology correctly reconstruct a known, underlying theory given the available data?*

First step is to define the "underlying law" you are trying to recover.

Level 0 Closure Test
--------------------

- replace central values of data with theory predictions from `closure_test_pdf`, set in the runcard.

Level 1 Closure Test
--------------------

- add randon fluctuations or "noise", which is sampled from a Gaussian distribution with central values your Level 0 data and s.d. the experimental uncertainties.


Model-specific Closure Test
---------------------------

