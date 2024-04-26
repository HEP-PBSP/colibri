The following are needed for the documentation set-up:

- conda install sphinx
- conda install sphinxcontrib-bibtex
- conda install sphinx-book-theme

(See e.g. https://sphinx-themes.org/ for other choices of themes)


From within the sphinx directory:

- sphinx-build -b html source build
- or 'make html'


Then 'open build/html/index.html' will open the webpage.


Important files:

- sphinx/source/conf.py configures the set-up.  The project name, authors, version as well as extensions and project theme can all be changed here.
- sphinx/source/index.rst is the 'front page', and table of contents etc are specified here
- sphinx/source/simunet_analysis.rst contains some first basic examples of how we can generate documentation directly from the code docstrings.


See also https://www.sphinx-doc.org/en/master/index.html for much more info on Sphinx
