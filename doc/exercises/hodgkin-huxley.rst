Numerical integration of the HH model of the squid axon
=======================================================

**Book chapters**

See `Chapter 2 Section 2 <Chapter_>`_ on general information about
the Hodgkin-Huxley equations and models.

.. _Chapter: http://neuronaldynamics.epfl.ch/online/Ch2.S2.html


**Python classes**

The :mod:`.hodgkin_huxley.HH` module contains all code required for this exercise.
At the beginning of your exercise solutions, import the contained functions by running

.. code-block:: py

	from neurodynex.hodgkin_huxley.HH import *

You can then simply run the exercise functions by executing

.. code-block:: py

	HH_Step()  # example Step-current injection
	HH_Sinus()  # example Sinus-current injection
	HH_Ramp()  # example Ramp-current injection

Exercise
--------

Use the function :func:`.HH_Step` to simulate a HH
neuron stimulated by a current step of a given amplitude. The goal of
this exercise is to modify the provided python functions and use the
``numpy`` and ``matplotlib`` packages to answer the following questions.

Question
~~~~~~~~

What is the lowest step current amplitude for generating at least one spike? Hint: use binary search on ``I_amp``, with a :math:`0.1\mu A` resolution.

Question
~~~~~~~~

What is the lowest step current amplitude to generate repetitive firing?

Question
~~~~~~~~

Look at :func:`.HH_Step` for ``I_amp = -5`` and ``I_amp = -1``. What is happening here? To which gating variable do you attribute this rebound spike?

Exercise
--------

Use the function :func:`.HH_Ramp` to simulate a HH neuron stimulated by a ramping curent.

Question
~~~~~~~~

What is the minimum current required to make a spike when the current is slowly increased (ramp current waveform) instead of being increased suddenly?

Exercise
--------

To solve this exercise, you will need to change the actual implementation of the model. Download directly the source file `HH.py <https://raw.githubusercontent.com/EPFL-LCN/neuronaldynamics-exercises/master/neurodynex/hodgkin_huxley/HH.py>`_. When starting Python in the directory containing the downloaded file, you run functions from it directly as follows:

.. code-block:: py
	
	import HH  # import the HH module, i.e. the HH.py file
	HH.HH_Step()  # access the LIF_Step function in HH.py

Then use any text editor to make changes in the ``HH.py`` file. 

**Hint**: If you are using iPython, you will have to reload the module after making changes, by typing:

.. code-block:: py
	
	reload(HH)

Question
~~~~~~~~

What is the current threshold for repetitive spiking if the density of sodium channels is increased by a factor of 1.5? To solve this, change the maximum conductance of sodium channel in :func:`.HH_Neuron`.