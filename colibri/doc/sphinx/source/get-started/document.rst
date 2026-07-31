.. _document:

Contributing to the Documentation
=================================

Colibri's documentation is written in `sphinx <http://www.sphinx-doc.org/en/master/>`__ .
You can find details on how to format with sphinx in `their website <https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`__.

Before pushing changes to the documentation, it is useful to compile them using:

.. code-block:: bash

    make html

in the ``colibri/doc/sphinx/`` directory, and make sure that the compiled version
looks as you expect it to, by opening it with:

.. code-block:: bash

    open build/html/index.html

Thanks!

