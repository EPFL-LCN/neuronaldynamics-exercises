.. _exercises-setup:

Setting up Python and Brian
===========================

To solve the exercises you need to install Python, Brian2 and the neurodynex package. The installation procedure we described here focuses on the tools we use in the classroom sessions at EPFL. For that reason we additionally set up a **conda environment** (which we call bmnn below) and install `Jupyter <http://jupyter.readthedocs.io/en/latest/install.html>`__ .



.. _exercises-setup-conda:

Using miniconda
---------------

We offer anaconda packages for the most recent releases, which is the easiest way of running the exercises. (Alternatively you can clone the sources from github)

Head over to the `miniconda download page <http://conda.pydata.org/miniconda.html>`__ and install **miniconda** (for Python 2.7 preferably).

Now execute the following commands to **install** the exercise package as well as **Brian2** and some dependencies. Note: we create a `conda environment <http://conda.pydata.org/docs/test-drive.html#managing-envs>`_ called 'bmnn'. You may want to change that name. In the last command we install `Jupyter <http://jupyter.org>`_ , a handy tool to create solution documents.


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


   Once you've create a new notebook, copy-paste the code of exercise one into the notebook and run it. Note that the first time you do this, the execution may take a little longer and, in some cases, you may see compilation warnings.

.. figure:: exc_images/StartJupyter_2.png
   :align: center

We recommend you to create one notebook per exercise.

.. note::

   	**Trouble shooting:** You may get errors like 'No module named 'neurodynex'. This is the case when your jupyter notebook does not see the packages you've just installed. As a solution, try to re-install jupyter **within** the environment:
   	.. code-block::

   	   	>> source activate bmnn
   	   	>> conda install jupyter


Links
-----
Here are some useful links to get started with Python and Brian

* `<https://www.python.org/about/gettingstarted/>`_
* `<https://brian2.readthedocs.io/en/latest/index.html>`_
* `<http://www.scipy.org>`_
* `<http://Matplotlib.sf.net>`_