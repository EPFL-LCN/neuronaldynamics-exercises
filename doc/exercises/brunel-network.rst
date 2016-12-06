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
    from neurodynex.tools import plot_tools
    import brian2 as b2

    rate_monitor, spike_monitor, voltage_monitor, monitored_spike_idx = LIF_spiking_network.simulate_brunel_network(sim_time=250. * b2.ms)
    plot_tools.plot_network_activity(rate_monitor, spike_monitor, voltage_monitor, spike_train_idx_list=monitored_spike_idx, t_min=0.*b2.ms)


Note that you can change all parameters of the neuron by using the named parameters of the function :func:`.simulate_brunel_network`. If you do not specify any parameter, the default values are used (see next code block). You can access these variables in your code by prefixing them with the module name (for example LIF_spiking_network.POISSON_INPUT_RATE).

.. code-block:: py

    # Default parameters of a single LIF neuron:
    V_REST = 0. * b2.mV
    V_RESET = +10. * b2.mV
    FIRING_THRESHOLD = +20. * b2.mV
    MEMBRANE_TIME_SCALE = 20. * b2.ms
    ABSOLUTE_REFRACTORY_PERIOD = 2.0 * b2.ms

    # Default parameters of the network
    SYNAPTIC_WEIGHT_W0 = 0.1 * b2.mV  # note: w_ee=w_ie = w0 and = w_ei=w_ii = -g*w0
    RELATIVE_INHIBITORY_STRENGTH_G = 4.  # balanced
    CONNECTION_PROBABILITY_EPSILON = 0.1
    SYNAPTIC_DELAY = 1.5 * b2.ms
    POISSON_INPUT_RATE = 12. * b2.Hz
    N_POISSON_INPUT = 1000


Exercise: model parameters and threshold rate
---------------------------------------------

In the first question, we get familiar with the model and parameters. Make sure you have read the `book chapter <http://neuronaldynamics.epfl.ch/online/Ch13.S4.html>`_ . Then have a look at the documentation of :func:`.simulate_brunel_network`. Note that in our implementation, the number of excitatory presynaptic poisson neurons (input from the external population) is a parameter `N_extern` and thus independent of `CE`.


Question:
~~~~~~~~~

* Run the simulation with the default parameters (see code block above). In that default configuration, what values  take the variables :math:`N_E`, :math:`N_I`, :math:`C_E`, :math:`C_I`, :math:`w_{EE}`, :math:`w_{EI}`, :math:`w_{IE}`, and :math:`w_{II}`? The variables are described in the book and in `Fig. 13.6 <http://neuronaldynamics.epfl.ch/online/Ch13.S4.html>`_

* What are the units of the weights w?

* The frequency :math:`\nu_{threshold}` is is the poisson rate of the external population sufficient to drive the neurons in the network to the firing threshold. Using Eq. :eq:`eq_nu_threshold`, compute :math:`\nu_{threshold}`. You can do this in Python, e.g. use LIF_spiking_network.FIRING_THRESHOLD for :math:`u_{thr}`, etc.


* Refering to Figure 13.7, left panel, what is the meaning of the value 1 on the y-axis (Input). What is the horizontal dashed line designating? How is it related to :math:`u_{thr}`?


* Run a simulation  for 500ms. Set `poisson_input_rate` to :math:`\nu_{threshold}`. Plot the network activity in the time interval [0ms, 500ms]. Is the network quiet (Q)?

* During the simulation time, what is the average firing rate of a single neuron? You can access the total number of spikes from the Brian2.SpikeMonitor: spike_monitor.num_spikes and the number of neurons in the network from spike_monitor.source.N .

.. math::
   :label: eq_nu_threshold

    \nu_{threshold} = \frac{u_{thr}}{N_{extern} w_{0} \tau_m}



Exercise: Population activity
-----------------------------

The network of spiking LIF-neurons shows characteristic population activities. In this exercise we investigate the patterns asynchronous irregular (AI), synchronous regular (SR), fast synchronous irregular (SI fast) and slow synchronous irregular (SI slow).

Question:
~~~~~~~~~

* The function :func:`.simulate_brunel_network` gives you three options to vary the input strength (y-axis in figure 13.7, a). What options do you have?

* Define a network of 6000 excitatory and 1500 inhibitory neurons. Find the appropriate parameters and simulate the network in the regimes AI, SR, SI-fast and SI-slow. For each of the four configurations, plot the network activity and compute the average firing rate. Run each simulation for at least 800ms and plot two figures for each simulation: one showing the complete simulation time and one showing only the last 50ms.

* What is the  population activity A(t) in each of the four conditions (in Hz, averaged over the last 200ms of your simulation)?

* Access the spike-monitor and calculate the interspike-intervals. Display the ISI distribution in a histogram for each of the four conditions.

* From the ISI, compute the Coefficient Of Variation (CV) for each condition. Read `Chapter 7.3.1 <http://neuronaldynamics.epfl.ch/online/Ch7.S3.html>`_ to learn more about the CV.


Exercise: Emergence of Synchronization
--------------------------------------
The different regimes emerge from from the recurrence and the relative strength of inhibition `g`. In the absence of recurrent feedback from the network, the network would approach a constant mean activity A(t).


Question:
~~~~~~~~~
* Simulate a network of 6000 excitatory and 1500 inhibitory neurons. Set the following parameters: poisson_rate = 35*b2.Hz, g=7.8. In which state is this network?

* What would be the population activity caused by the external input only? We can simulate this. Run a simulation of the same network, but disable the recurrent feedback: simulate_brunel_network(...,w0=0.*b2.mV, w_external = LIF_spiking_network.SYNAPTIC_WEIGHT_W0).

* Explain why the non-recurrent network shows a strong synchronization in the beginning and why this synchronization fades out.

* The non recurrent network is strongly synchronized in the beginning. Is the connected network simply "locked" to this initial synchronization? You can falsify this hypothesis by initializing each neuron in the network with a random vm. Run the simulation with `random_vm_init=True` to see how the synchronization emerges over time (see figure below).


.. figure:: exc_images/Brunel_Synchronization.png
   :align: center

   Simulation of a network with random v_m initialization. The synchronization of the neurons is not a residue of shared initial conditions, but emerges over time.
