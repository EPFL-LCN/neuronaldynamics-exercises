Hopfield Network model of associative memory
============================================

**Book chapters**

See `Chapter 17 Section 2 <Chapter17_>`_ for an introduction to Hopfield networks.

.. _Chapter17: http://neuronaldynamics.epfl.ch/online/Ch17.S2.html

**Python classes**

Hopfield networks can be analyzed mathematically. In this Python exercise we focus on visualization and simulation to develop our intuition about Hopfield dynamics.

We provide a couple of functions to easily create patterns, store them in the network and visualize the network dynamics. Check the modules :mod:`.hopfield_network.network`, :mod:`.hopfield_network.pattern_tools` and :mod:`.hopfield_network.plot_tools` to learn the building blocks we provide.

.. note::
    If you instantiate a new object of class  :class:`network.HopfieldNetwork` it's default dynamics are **deterministic** and **synchronous**. That is, all states are updated at the same time using the sign function. We use this dynamics in all exercises described below.


Getting started:
----------------
Run the following code. Read the inline comments and check the documentation. The patterns and the flipped pixels are randomly chosen. Therefore the result changes every time you execute this code. Run it several times and change some parameters like nr_patterns and nr_of_flips.

.. code-block:: python

    %matplotlib inline
    from neurodynex.hopfield_network import network, pattern_tools, plot_tools

    pattern_size = 5

    # create an instance of the class HopfieldNetwork
    hopfield_net = network.HopfieldNetwork(nr_neurons= pattern_size**2)
    # instantiate a pattern factory
    factory = pattern_tools.PatternFactory(pattern_size, pattern_size)
    # create a checkerboard pattern and add it to the pattern list
    checkerboard = factory.create_checkerboard()
    pattern_list = [checkerboard]

    # add random patterns to the list
    pattern_list.extend(factory.create_random_pattern_list(nr_patterns=3, on_probability=0.5))
    plot_tools.plot_pattern_list(pattern_list)
    # how similar are the random patterns and the checkerboard? Check the overlaps
    overlap_matrix = pattern_tools.compute_overlap_matrix(pattern_list)
    plot_tools.plot_overlap_matrix(overlap_matrix)

    # let the hopfield network "learn" the patterns. Note: they are not stored
    # explicitly but only network weights are updated !
    hopfield_net.store_patterns(pattern_list)

    # create a noisy version of a pattern and use that to initialize the network
    noisy_init_state = pattern_tools.flip_n(checkerboard, nr_of_flips=4)
    hopfield_net.set_state_from_pattern(noisy_init_state)

    # from this initial state, let the network dynamics evolve.
    states = hopfield_net.run_with_monitoring(nr_steps=4)

    # each network state is a vector. reshape it to the same shape used to create the patterns.
    states_as_patterns = factory.reshape_patterns(states)
    # plot the states of the network
    plot_tools.plot_state_sequence_and_overlap(states_as_patterns, pattern_list, reference_idx=0, suptitle="Network dynamics")



.. figure:: exc_images/HF_CheckerboardAndPatterns.png
    :align: center

    Six patterns are stored in a Hopfield network.

.. figure:: exc_images/HF_CheckerboardRecovered2.png
    :align: center

    The network is initialized with a (very) noisy pattern S(t=0). Then, the dynamics recover pattern P0 in 5 iterations.


.. note::
   The network state is a vector of N neurons. For visualization we use 2d patterns which are two dimensional numpy.ndarrays of size = (length, width). To store such patterns, initialize the network with N = length x width neurons.


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


Exercise: N=4x4 Hopfield-network
--------------------------------
We study how a network stores and retrieve patterns. Using a small network of only 16 neurons allows us to have a close look at the network weights and dynamics.


Question: Storing a single pattern
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Modify the Python code given above to implement this exercise:

#. Create a network with N=16 neurons.
#. Create a single 4 by 4 checkerboard pattern.
#. Store the checkerboard in the network.
#. Set the initial state of the network to a noisy version of the checkerboard (nr_flipped_pixels = 5).
#. Let the network dynamics evolve for 4 iterations.
#. Plot the sequence of network states along with the overlap of network state with the checkerboard.

Now test whether the network can still retrieve the pattern if we increase the number of flipped pixels. What happens at nr_flipped_pixels = 8, what if nr_flipped_pixels > 8 ?

Question: the weights matrix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The patterns a Hopfield network learns are not stored explicitly. Instead, the network learns by adjusting the weights to the pattern set it is presented during learning. Let's visualize this.

#. Create a new 4x4 network. Do not yet store any pattern.
#. What is the size of the network matrix?
#. Visualize the weight matrix using the function :func:`.plot_tools.plot_nework_weights`. It takes the network as a parameter.
#. Create a checkerboard, store it in the network.
#. Plot the weights matrix. What weight values do occur?
#. Create a new 4x4 network
#. Create an L-shaped pattern (look at the pattern factory doc), store it in the network
#. Plot the weights matrix. What weight values do occur?
#. Create a new 4x4 network
#. Create a checkerboard and an L-shaped pattern. Store **both** patterns in the network
#. Plot the weights matrix. What weight values do occur? How does this matrix compare to the two previous matrices?


.. note::

    The mapping of the 2 dimensional patterns onto the one dimensional list of network neurons is internal to the implementation of the network. You cannot know which pixel (x,y) in the pattern corresponds to which network neuron i.


Question (optional): Weights Distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
It's interesting to look at the weights distribution in the three previous cases. You can easily plot a histogram by adding the following two lines to your script. It assumes you have stored your network in the variable 'hopfield_net'.

.. code-block:: py

    plt.figure()
    plt.hist(hopfield_net.weights.flatten())


Exercise: Capacity of an N=100 Hopfield-network
-----------------------------------------------
Larger networks can store more patterns. There is a theoretical limit: the capacity of the Hopfield network. Read `chapter "17.2.4 Memory capacity" <http://neuronaldynamics.epfl.ch/online/Ch17.S2.html>`_ to learn how memory retrieval, pattern completion and the network capacity are related.

Question:
~~~~~~~~~
A Hopfield network implements so called **associative** or **content-adressable** memory. Explain what this means.

Question:
~~~~~~~~~
Using the value :math:`C_{store}` given in the book, how many patterns can you store in a N=10x10 network? Use this number **K** in the next question:

Question:
~~~~~~~~~
Create an N=10x10 network and store a checkerboard pattern together with **(K-1) random patterns**. Then initialize the network with the **unchanged** checkerboard pattern. Let the network evolve for five iterations.

Rerun your script a few times. What do you observe?


Exercise: Non-random patterns
-----------------------------

In the previous exercises we used random patterns. Now we us a list of structured patterns: the letters A to Z. Each letter is represented in a 10 by 10 grid.

.. figure:: exc_images/HF_LetterAandOverlap.png
   :align: center

   Eight letters (including 'A') are stored in a Hopfield network. The letter 'A' is not recovered.


Question:
~~~~~~~~~
Run the following code. Read the inline comments and look up the doc of functions you do not know.

.. code-block:: Py

    %matplotlib inline
    import matplotlib.pyplot as plt
    from neurodynex.hopfield_network import network, pattern_tools, plot_tools
    import numpy

    # the letters we want to store in the hopfield network
    letter_list = ['A', 'B', 'C', 'S', 'X', 'Y', 'Z']

    # set a seed to reproduce the same noise in the next run
    # numpy.random.seed(123)

    abc_dictionary =pattern_tools.load_alphabet()
    print("the alphabet is stored in an object of type: {}".format(type(abc_dictionary)))
    # access the first element and get it's size (they are all of same size)
    pattern_shape = abc_dictionary['A'].shape
    print("letters are patterns of size: {}. Create a network of corresponding size".format(pattern_shape))
    # create an instance of the class HopfieldNetwork
    hopfield_net = network.HopfieldNetwork(nr_neurons= pattern_shape[0]*pattern_shape[1])

    # create a list using Pythons List Comprehension syntax:
    pattern_list = [abc_dictionary[key] for key in letter_list ]
    plot_tools.plot_pattern_list(pattern_list)

    # store the patterns
    hopfield_net.store_patterns(pattern_list)

    # # create a noisy version of a pattern and use that to initialize the network
    noisy_init_state = pattern_tools.get_noisy_copy(abc_dictionary['A'], noise_level=0.2)
    hopfield_net.set_state_from_pattern(noisy_init_state)

    # from this initial state, let the network dynamics evolve.
    states = hopfield_net.run_with_monitoring(nr_steps=4)

    # each network state is a vector. reshape it to the same shape used to create the patterns.
    states_as_patterns = pattern_tools.reshape_patterns(states, pattern_list[0].shape)

    # plot the states of the network
    plot_tools.plot_state_sequence_and_overlap(
        states_as_patterns, pattern_list, reference_idx=0, suptitle="Network dynamics")


Question:
~~~~~~~~~
Add the letter 'R' to the letter list and store it in the network. Is the pattern 'A' still a fixed point? Does the overlap between the network state and the reference pattern 'A' always decrease?

Question:
~~~~~~~~~
Make a guess of how many letters the network can store. Then create a (small) set of letters. Check if **all** letters of your list are fixed points under the network dynamics. Explain the discrepancy between the network capacity C (computed above) and your observation.


Exercise: Bonus
---------------
The implementation of the Hopfield Network in :mod:`.hopfield_network.network` offers a possibility to provide a custom update function :meth:`.HopfieldNetwork.set_dynamics_to_user_function`. Have a look at the source code of :meth:`.HopfieldNetwork.set_dynamics_sign_sync` to learn how the update dynamics are implemented. Then try to implement your own function. For example, you could implement an asynchronous update with stochastic neurons.
