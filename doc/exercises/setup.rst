.. _exercises-setup:

Setting up Python and Brian
===========================

To solve the exercises you need to install Python, Brian2 and the neurodynex package. The installation procedure we described here focuses on the tools we use in the classroom sessions at EPFL. For that reason we additionally set up a **conda environment** (which we call bmnn below) and install `Jupyter <http://jupyter.readthedocs.io/en/latest/install.html>`__ .



.. _exercises-setup-conda:

Using miniconda (recommended)
-----------------------------

We offer anaconda packages for the most recent releases, which is the easiest way of running the exercises.

Head over to the `miniconda download page <http://conda.pydata.org/miniconda.html>`__ and install **miniconda** (for Python 2.7 preferably).

Now execute the following commands to **install** the exercise package as well as **Brian2** and some dependencies. Note: we create a `conda environment <http://conda.pydata.org/docs/test-drive.html#managing-envs>`_ called 'bmnn'. You may want to change that name. In the last command, we install `Jupyter <http://jupyter.org>`_, a handy tool to create solution documents.


.. code-block:: bash

    >> conda create --name bmnn python=2.7
    >> source activate bmnn
    >> conda install -c brian-team -c epfl-lcn neurodynex
    >> conda install jupyter


If you need to  **update** the exercise package, call:

.. code-block:: bash

    >> source activate bmnn
    >> conda update -c brian-team -c epfl-lcn neurodynex


You now have the tools you need to solve the python exercises. To get started, open a terminal, move to the folder where you want your code being stored and start a Jupyter notebook:

.. code-block:: bash

    >> cd your_folder
    >> source activate bmnn
    >> jupyter notebook

.. figure:: exc_images/StartJupyter.png
   :align: center

   Starting Jupyter will open your browser. Select NEW, Python2 to get a new notebook page. Depending on what else you have installed on your computer, you may have to specify the kernel. In the case shown here, it's the Python-bmnn installation.


   Once you've create a new notebook, copy-pase the code of exercise one into the notebook and run it. Note that the first time you do this, the execution may take a little longer and, in some cases, you may see compilation warnings.

.. figure:: exc_images/StartJupyter_2.png
   :align: center

We recommend you to create one notebook per exercise.

.. note::

   	**Trouble shooting:** You may get errors like 'No module named 'neurodynex'. This is the case when your jupyter notebook does not see the packages you've just installed. As a solution, try to re-install jupyter **within** the environment:
   	.. code-block::

   	   	>> source activate bmnn
   	   	>> conda install jupyter


Alternative procedure: Using python and pip
-------------------------------------------

If you already have Python installed and prefer using PIP, you can get the most recent versions of this repository as a `pypi package <https://pypi.python.org/pypi/neurodynex/>`__ called ``neurodynex``.

To install the exercises using ``pip`` simply execute (the ``--upgrade`` flag will overwrite existing installations with the newest versions):

.. code-block:: bash

   pip install --upgrade jupyter
   pip install --upgrade neurodynex

.. note::

   	Should you want to run `Spyder <https://github.com/spyder-ide/spyder>`_ to work on the exercises, and you're running into problems (commonly, after running ``conda install spyder`` you can not start ``spyder`` due to an error related to numpy), try the following:

   	.. code-block:: bash

   		# create a new conda environment with spyder and the exercises
   		conda create --name neurodynex -c brian-team -c epfl-lcn neurodynex spyder

   		# activate the environment
   		source activate neurodynex

   	This creates a new conda environment (`here is more information on conda environments <http://conda.pydata.org/docs/test-drive.html#managing-envs>`_) in which you can use spyder together with the exercises.


Links
-----
Here are some useful links to get started with Python and Brian

* `<https://www.python.org/about/gettingstarted/>`_
* `<https://brian2.readthedocs.io/en/latest/index.html>`_
* `<http://www.scipy.org>`_
* `<http://Matplotlib.sf.net>`_