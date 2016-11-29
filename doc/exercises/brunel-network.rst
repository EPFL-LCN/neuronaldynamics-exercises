Network of LIF neurons
======================

In this exercise we study a well known network of sparsely connected Leaky-Integrate-And-Fire neurons (Brunel, 2000).

**Book chapters**

The Brunel model is introduced in `Chapter 13 Section 4.2 <http://neuronaldynamics.epfl.ch/online/Ch13.S4.html>`_ . The network structure is shown in figure 13.6b. Read the section "Synchrony, oscillations, and irregularity" and have a look at Figure 13.7. For this exercise, you can skip the explanations related to the Fokker-Planck equation.


**Python classes**

The module :mod:`.brunel_model.LIF_spiking_network` implements a parametrized network. The figure below shows the simulation result of a default configuration.


.. figure:: exc_images/Brunel_Spiking_LIF.png
   :align: center

   Simulation result. Top: raster plot of 150 randomly selected neurons. Three spike trains are visually highlighted. Middle: time evolution of the population activity A(t). Bottom: Membrane voltage of three neurons. The red color in the top and bottom panels identifies the same neuron.


To get started, call the function  :func:`LIF_spiking_network.getting_started` or copy the following code into a Jupyter notebook.


.. code-block:: py

    %matplotlib inline
    from neurodynex.brunel_model import LIF_spiking_network
    import brian2 as b2

    rate_monitor, spike_monitor, voltage_monitor, monitored_spike_idx = LIF_spiking_network.simulate_brunel_network(N_Excit=2000, sim_time=100. * b2.ms, monitored_subset_size=150)
    LIF_spiking_network.plot_network_activity(rate_monitor, spike_monitor, voltage_monitor, spike_train_idx_list=monitored_spike_idx)




Note that you can change all parameters of the neuron by using the named parameters of the function :func:`.simulate_brunel_network`. If you do not specify any parameter, the default values are used (see next code block). You can access these variables in your code by prefixing them with the module name (for example LIF_spiking_network.POISSON_INPUT_RATE).

.. code-block:: py

    V_REST = 0. * b2.mV
    V_RESET = +10. * b2.mV
    FIRING_THRESHOLD = +20. * b2.mV
    MEMBRANE_TIME_SCALE = 20. * b2.ms
    ABSOLUTE_REFRACTORY_PERIOD = 2.0 * b2.ms
    # Default parameters of the network
    SYNAPTIC_WEIGHT_W0 = 0.1 * b2.mV
    # note: w_ee = w_ei = w0 and w_ie=w_ii = -g*w0
    RELATIVE_INHIBITORY_STRENGTH_G = 4.  # balanced
    CONNECTION_PROBABILITY_EPSILON = 0.1
    SYNAPTIC_DELAY = 1.5*b2.ms
    POISSON_INPUT_RATE = 12. * b2.Hz
    N_POISSON_INPUT = 1000


Exercise: model parameters and threshold rate
---------------------------------------------

In the first question, we get familiar with the model and parameters. Make sure you have read the book chapters. Then have a look at the documentation of :func:`.simulate_brunel_network`. Note that in our implementation, the number of excitatory presynaptic poisson neurons (input from the external population) is a parameter (independent of the size of the excitatory population).


Question:
~~~~~~~~~
Run the simulation with the default parameters (see code block above). For this default configuration, what values  take the following variables :

* CE, CI, w_EI, w_II

* What are the units of the weights w_EI and w_II?

* The frequency nu_threshold is is the poisson rate for which the external population drives the neurons to the firing threshold.  equation TODO. Compute nu_threshold.

* What is the meaning of the value 1 on the y-axis (Input); what is g (the x-axis)?

* Refering to Figure 13.7, left panel, what is the horizontal dashed line designating? How is it related to nu_threshold?


Exercise: Population activity
-----------------------------

The network of spiking LIF-neurons shows characteristic population activities. In this exercise we investigate the patterns asynchronous irregular (AI), synchronous regular (SR), fast synchronous irregular (SI fast) and slow synchronous irregular (SI slow).


Question:
~~~~~~~~~

TODO

* The function :func:`.simulate_brunel_network` gives you two options to vary the input strength (y-axis in figure 13.7, a). What options do you have?

* using a network of 5000 excitatory neurons, find the appropriate parameters and simulate the network in the regimes AI, SR, SI-fast and SI-slow.



* In SR, what is the average firing frequency of a single neuron? Use the spike monitor to ...

* Simulate the network with only 2000 neurons in SR regime, what do you expect to happen?
