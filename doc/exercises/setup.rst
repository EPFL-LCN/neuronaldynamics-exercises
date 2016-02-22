.. _exercises-setup:

Setting up Python and Brian
===========================

Using python and pip
--------------------

We provide the most recent versions of this repository as a `pypi package <https://pypi.python.org/pypi/neurodynex/>`__ called ``neurodynex``.

To install the exercises using ``pip`` simply execute (the ``--upgrade`` flag will overwrite existing installations with the newest versions):

.. code-block:: bash

   pip install --upgrade neurodynex


Using anaconda/miniconda
---------------------------

We offer anaconda packages for the most recent releases, which is the easiest way of running the exercises.

Head over to the `miniconda download page <http://conda.pydata.org/miniconda.html>`__ and install **miniconda** (for Python 2.7 preferably). To **install or update** the exercise classes for your anaconda environment, it suffices to run:

.. code-block:: bash

   conda install -c brian-team -c epfl-lcn neurodynex