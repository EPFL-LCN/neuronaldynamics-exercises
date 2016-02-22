.. _exercises-setup:

Setting up Python and Brian
===========================

Using python and pip
--------------------

We provide the most recent versions of this repository as a `pypi package <https://pypi.python.org/pypi/neurodynex/>`__ called ``neurodynex``.

To install the exercises using ``pip`` simply execute (the ``--upgrade`` flag will overwrite existing installations with the newest versions):

.. code-block:: bash

   pip install --upgrade neurodynex


Using anaconda or miniconda
---------------------------

Temporarily, we do not offer anaconda packages, however, you can easily install neurodynex by executing the following:

.. code-block:: bash
	
   conda install conda-build
   conda config --add channels https://conda.anaconda.org/brian-team
   
   conda skeleton pypi neurodynex
   conda build neurodynex
   
