.. _exercises-setup:

Setting up Python and Brian
===========================

Using python and pip
--------------------

We provide the most recent versions of this repository as a `pypi package <https://pypi.python.org/pypi/neurodynex/>`__ called ``neurodynex``.

To install the exercises using ``pip`` simply execute (the ``--upgrade`` flag will overwrite existing installations with the newest versions):

.. code-block:: bash

   pip install --upgrade neurodynex


.. _exercises-setup-conda:

Using anaconda/miniconda
------------------------

We offer anaconda packages for the most recent releases, which is the easiest way of running the exercises.

Head over to the `miniconda download page <http://conda.pydata.org/miniconda.html>`__ and install **miniconda** (for Python 2.7 preferably). To **install or update** the exercise classes for your anaconda environment, it suffices to run:

.. code-block:: bash

   conda install -c brian-team -c epfl-lcn neurodynex

.. note::

   	Should you want to run `Spyder <https://github.com/spyder-ide/spyder>`_ to work on the exercises, and you're running into problems (commonly, after running ``conda install spyder`` you can not start ``spyder`` due to an error related to numpy), try the following:

   	.. code-block:: bash

   		# create a new conda environment with spyder and the exercises
   		conda create --name neurodynex -c brian-team -c epfl-lcn neurodynex spyder

   		# activate the environment
   		source activate neurodynex

   	This creates a new conda environment (`here is more information on conda environments <http://conda.pydata.org/docs/test-drive.html#managing-envs>`_) in which you can use spyder together with the exercises.

