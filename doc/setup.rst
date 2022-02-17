.. _setup:

Installation
===========================

To solve the exercises you need to install ``Python``, the ``Brian2`` simulator and the ``neurodynex3`` package. The tutorial below will explain two different ways of installing everything:

- using the :ref:`pip <setup-pip>` package manager (Linux, macOS)

- using the :ref:`conda <setup-conda>` package manager, if you have Anaconda installed (Linux, macOS, Windows).

In both cases, we will use the package manager to create a virtual environment called ``bmnn`` and install everything there.



.. _setup-pip:

Using ``pip``
--------------
1. We will start by creating a virtual environemt that contains a separate Python installation. To do this, we first need the ``virtualenv`` package. Install this by running:

   .. code-block:: bash

      >> pip install virtualenv

2. Now create the virtual environment in a folder called ``bmnn``:

   .. code-block:: bash

      >> virtualenv bmnn

3. Activate the virtual environment:

   .. code-block:: bash

      >> source bmnn/bin/activate

4. Now install ``neurodynex3`` and all its requirements:

   .. code-block:: bash
   
      >> pip install --upgrade neurodynex3
      >> pip install jupyter

5. You can now use Python in this environment as you normally would. Move to the folder where you want your code to be stored and start a Jupyter notebook by running:

   .. code-block:: bash

      >> cd your_folder
      >> jupyter notebook

6. Finally, when you are done using the virtual environment, you can deactivate it:

   .. code-block:: bash

      >> deactivate

.. note::

   If something goes wrong inside the virtual environment, you can simply delete it by removing its folder (with the regular ``rm -r`` command) and start over:

   .. code-block:: bash

      >> deactivate
      >> rm -r bmnn



.. _setup-conda:

Using ``conda``
----------------
`Anaconda <https://www.anaconda.com/distribution/>`_ is a Python distribution that can be installed on Linux, macOS, and Windows. It comes together with a package manager called ``conda``. To run ``conda`` commands if you are using Windows, first start the ``Anaconda Prompt``.

.. image:: setup_images/anaconda-prompt.png
   :align: center

If you are using Linux or macOS, you can run ``conda`` commands in a regular terminal.

1. We start by creating a virtual environemt that contains a separate Python installation. The virtual environment is called ``bmnn``:

   .. code-block:: bash

      >> conda create --name bmnn python

2. Activate the virtual environment:

   .. code-block:: bash

      >> conda activate bmnn

3. Now install required Python packages:

   .. code-block:: bash

      >> conda install numpy scipy jupyter matplotlib mpmath setuptools setuptools_scm mock nose

4. Install ``Brian2``:

   .. code-block:: bash

      >> conda install -c conda-forge brian2

5. We will now install ``neurodynex3``. **Note:** this step is done using ``pip``, **not** ``conda``. First make sure that you are using ``pip`` inside the virtual environment:

   .. code-block:: bash

      >> pip --version
      pip 20.0.2 from //anaconda3/envs/bmnn/.../pip (python 3.8)

6. Now run the install command:

   .. code-block:: bash

      >> pip install neurodynex3

7. You can now use Python in this environment as you normally would. Move to the folder where you want your code to be stored and start a Jupyter notebook by running:

   .. code-block:: bash

      >> cd your_folder
      >> jupyter notebook

8. Finally, when you are done using the virtual environment, you can deactivate it:

   .. code-block:: bash

      >> conda deactivate

.. note::

   If something goes wrong inside the virtual environment, you can simply delete it and start over:

   .. code-block:: bash

      >> conda deactivate
      >> conda remove --name bmnn --all
   
   More information can be found in the `conda documentation <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_.



.. _setup-jupyter:

Start a Jupyter notebook
--------------------------

1. First, activate the virtual environment. If you use ``pip``, activate the virtual environment with

   .. code-block:: bash

      >> source bmnn/bin/activate

   If you use ``conda``, activate the virtual environment with:

   .. code-block:: bash

      >> conda activate bmnn

   .. note::
   
      Always make sure you use programs that are inside the virtual environment. To see that you are using the ``jupyter`` that is inside the ``bmnn`` virtual environment on Linux/macOS, you can use the ``which`` command

      .. code-block:: bash

         >> which jupyter
         .../bmnn/bin/jupyter

      and on Windows you can use the ``where`` command

      .. code-block:: doscon

         > where jupyter
         C:\...\Anaconda3\envs\bmnn\jupyter.exe

2. Move to the folder where you want your code to be stored and start a Jupyter notebook:

   .. code-block:: bash

      >> cd your_folder
      >> jupyter notebook

3. Starting Jupyter will open your browser. Select ``New``, ``Python3`` to get a new notebook page. Depending on what else you have installed on your computer, you may have to specify the kernel.

   .. figure:: setup_images/start-notebook.png
      :align: center
      :target: _images/start-notebook.png

4. Once you have created a new notebook, copy-paste the code of the exercise into the notebook and run it. Note that the first time you do this, the execution may take a little longer and, in some cases, you may see compilation warnings.

   .. figure:: setup_images/run-code.png
      :align: center
      :target: _images/run-code.png

We recommend you to create one notebook per exercise.


Links
-----
Here are some useful links to get started with Python and Brian:

- `Python documentation <https://www.python.org/doc>`_
- `Brian2 documentation <https://brian2.readthedocs.io/en/stable>`_
- `Jupyter notebook documentation <https://jupyter-notebook.readthedocs.io/en/stable>`_
- `Matplotlib documentation <https://matplotlib.org/tutorials/index.html>`_
- `conda documentation <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_
