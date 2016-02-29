Hopfield Network model of associative memory
============================================

**Book chapters**

See `Chapter 17 Section 2 <Chapter17_>`_ for an introduction to Hopfield networks.

.. _Chapter17: http://neuronaldynamics.epfl.ch/online/Ch17.S2.html

**Python classes**

The :mod:`.hopfield_network.hopfield` module contains all code required for this exercise.
At the beginning of your exercise solutions, import the contained functions by running

.. code-block:: python

	from neurodynex.phase_plane_analysis.fitzhugh_nagumo import *

You can then simply run the exercise functions by executing the functions, e.g.

.. code-block:: python

	get_trajectory() 
	get_fixed_point()
	plot_flow()

Introduction: Hopfield-networks
-------------------------------

This exercise uses a model in which neurons are pixels and take the values of -1 (*off*) or +1 (*on*). The network can store a certain number of pixel patterns, which is to be investigated in this exercise. During a retrieval phase, the network is started with some initial configuration and the network dynamics evolves towards the stored pattern (attractor) which is closest to the initial configuration. 

The dynamics is that of equation:

.. math::

	S_i(t+1) = sgn\left(\sum_j w_{ij} S_j(t)\right)

In the Hopfield model each neuron is connected to every other neuron
(full connectivity). The connection matrix is

.. math:: 
	w_{ij} = \frac{1}{N}\sum_{\mu} p_i^\mu p_j^\mu

where N is the number of neurons, :math:`p_i^\mu` is the value of neuron
:math:`i` in pattern number :math:`\mu` and the sum runs over all
patterns from :math:`\mu=1` to :math:`\mu=P`. This is a simple
correlation based learning rule (Hebbian learning). Since it is not a
iterative rule it is sometimes called one-shot learning. The learning
rule works best if the patterns that are to be stored are random
patterns with equal probability for on (+1) and off (-1). In a large
networks (N to infinity) the number of random patterns that can be
stored is approximately 0.14 times N.


Exercise: 4x4 Hopfield-network
------------------------------

This exercise deals not only with Python functions, but with Python classes and objects. The class :class:`HopfieldNetwork <.hopfield_network.hopfield.HopfieldNetwork>` implements a Hopfield network. To run the exercises you will have to instantiate the network:

.. code-block:: python

    from neurodynex.hopfield_network.hopfield import HopfieldNetwork
    n = HopfieldNetwork(4)  # instantiates a new HopfieldNetwork

.. note::  
	See the :class:`documentation for the HopfieldNetwork class <.hopfield_network.hopfield.HopfieldNetwork>` to see all methods you can use on a instantiated HopfieldNetwork.

Storing patterns
~~~~~~~~~~~~~~~~

Create an instance of the :class:`HopfieldNetwork <.hopfield_network.hopfield.HopfieldNetwork>` with N=4. Use the :meth:`make_pattern <.hopfield_network.hopfield.HopfieldNetwork.run>` method to store a pattern (default is one random pattern with half of its pixels *on*) and test whether it can be retrieved with the :meth:`run <.hopfield_network.hopfield.HopfieldNetwork.run>` method:

.. code-block:: python

	n.run()  # Note: this will fail with a RuntimeError if no patterns have been stored before

The :meth:`run <.hopfield_network.hopfield.HopfieldNetwork.run>` method, by defaults, runs the dynamics for the first pattern with no pixel flipped.

Question: Capacity of the 4x4 network
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

What is the experimental maximum number of random patterns the 4x4 network is able to memorize? 

Store more and more random patterns and test retrieval of some of them. The first few patterns should be stored perfectly, but then the performance gets worse. 

Does this correspond to the theoretical maximum number of random patterns the network should be able to memorize?

Exercise: 10x10 Hopfield-network
--------------------------------

Question: Capacity of the 10x10 network
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Increase the network size to 10x10 and repeat the steps of the previous exercise.

Question: Error correction
~~~~~~~~~~~~~~~~~~~~~~~~~~

Instatiate a network and store a finite number of random patterns, e.g. 8. 

How many wrong pixels can the network tolerate in the initial state, such that it still settles into the correct pattern?

.. note::  
	See the documentation for the :meth:`run method <.hopfield_network.hopfield.HopfieldNetwork.run>` to see how to control which percentage of pixels is flipped.

Question: Storing alphabet letters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Try to store alphabetic characters as the relevant patterns. How good is the retrieval of patterns? What is the reason?

.. note::  
	See the documentation for the :meth:`make_pattern method <.hopfield_network.hopfield.HopfieldNetwork.make_pattern>` on how to store alphabet characters.

Exercise: Bonus
---------------

Try one of the preceding points in bigger networks.

Try `downloading the source code for the network <https://raw.githubusercontent.com/EPFL-LCN/neuronaldynamics-exercises/master/neurodynex/hopfield_network/hopfield.py>`_, and modify it by adding a smooth transfer function *g* to the neurons. A short introducion on how to run the downloaded file :ref:`can be found here <exercises-hh-downloading>`.