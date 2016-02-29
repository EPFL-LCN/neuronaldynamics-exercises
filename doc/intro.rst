Introduction
===================================

This repository contains python exercises accompanying the book
`Neuronal Dynamics <http://neuronaldynamics.epfl.ch/>`__ by Wulfram
Gerstner, Werner M. Kistler, Richard Naud and Liam Paninski.
References to relevant chapters will be added in the `Teaching Materials <http://neuronaldynamics.epfl.ch/lectures.html>`__ section of
the book homepage.

Quickstart
----------

See the indiviual :ref:`exercises <exercises-index>` - they contain instructions on how to use the python code to solve them.

To install the exercises using ``pip`` simply execute:

.. code-block:: bash

   pip install --upgrade neurodynex

To install the exercises with anaconda/miniconda execute: 

.. code-block:: bash

   conda install -c brian-team -c epfl-lcn neurodynex

See :ref:`the setup instructions <exercises-setup>` for details on how to install the python classes needed for the exercises. 

Brian1
------

We are currently rewriting the python exercises to use the more recent `Brian2 Simulator <https://github.com/brian-team/brian2>`__. The old brian1 exercises are available on the `brian1 branch <https://github.com/EPFL-LCN/neuronaldynamics-exercises/tree/brian1>`__.

Requirements
------------

The following requirements should be met:

-  Either Python 2.7 or 3.4
-  `Brian2 Simulator <https://github.com/brian-team/brian2>`__
-  Numpy
-  Matplotlib
-  Scipy (only required in some exercises)
