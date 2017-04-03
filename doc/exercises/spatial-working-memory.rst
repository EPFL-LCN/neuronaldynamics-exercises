Spatial Working Memory (Compte et. al.)
=======================================

In this exercise we study a model of spatial working memory. The model has been introduced by Compte et. al. [1]. The parameters used here differ from the original paper. They are changed such that we can study some effects in small networks. Using only 512 excitatory and 128 inhibitory neurons, we can still study the main mechanisms while saving simulation time.


**Book chapters**

Read the chapter ` Bump attractors and spontaneous pattern formation <http://neuronaldynamics.epfl.ch/online/Ch18.S3.html>`_ before doing this exercise. If you have access to a scientific library, you may also want to read the original publication [1].


**Python classes**

The module :mod:`.working_memory_network.wm_model` implements a parametrized network.


.. figure:: exc_images/WorkingMemory_Demo.png
   :align: center

   A weak stimulus, centered at 120deg, is applied to the network from t=200ms to t=400ms. This creates an activity bump bump in the excitatory subpopulation. The activity sustains after the end of the stimulation.


To get started, call the function  :func:`.working_memory_network.wm_model.getting_started` or copy the following code into a Jupyter notebook.


.. code-block:: py

    %matplotlib inline
    from neurodynex.brunel_model import LIF_spiking_network
    from neurodynex.tools import plot_tools
    import brian2 as b2

    rate_monitor, spike_monitor, voltage_monitor, monitored_spike_idx = LIF_spiking_network.simulate_brunel_network(sim_time=250. * b2.ms)
    plot_tools.plot_network_activity(rate_monitor, spike_monitor, voltage_monitor, spike_train_idx_list=monitored_spike_idx, t_min=0.*b2.ms)


.. code-block:: py

    # Default parameters of a single LIF neuron:
    V_REST = 0. * b2.mV
    sa;dlfkjas;l

Exercise: Spontanous Bump Formation
-----------------------------------
take book chapter,
show: higher poisson rate OR "flat" input lead to bump at
1. random location
2. constant width
change params (gaussian profile width) too narrow/large


Exercise: Decoding the population activity into a population vector
-------------------------------------------------------------------
In the raster plot (see figure (todo)) you see that the population of spiking neurons keeps a memory of the stimulus. In this exercise we decode that spiking activity into the population vector, i.e. the  angle stored in the working memory network at a given time t. To do so, we access the data in the  Brian2 SpikeMonitor returned by the simulation. Read the `Brian2 documentation <http://brian2.readthedocs.io/en/stable/user/recording.html>`_ to see how one can access spike trains.



Question:
~~~~~~~~~
asdfasdf

* Bla




**References**

[1] Compte, A., Brunel, N., Goldman-Rakic, P. S., & Wang, X. J. (2000). Synaptic mechanisms and network dynamics underlying spatial working memory in a cortical network model. Cerebral Cortex, 10(9), 910-923.

[2] This exercise is inspired by material from Stanford University, BIOE 332: Large-Scale Neural Modeling, Kwabena Boahen & Tatiana Engel, 2013, online available.
