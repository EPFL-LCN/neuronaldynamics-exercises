Perceptual Decision Making (Wong & Wang)
========================================

In this exercise we study decision making in a network of competing populations of spiking neurons. The network has been proposed by Wong and Wang in 2006 [1] as a model of decision making in a visual motion detection task. The decision making task and the network are described in the `book  <http://neuronaldynamics.epfl.ch/online/Ch16.html>`_ and in the original publication (see :ref:`location-references` [1]).


.. _location-phase_plane:

.. figure:: exc_images/DecisionMaking_PhasePlane_3.png
    :align: center
    :width: 100%

    Decision Space.
    Each point represents the firing rates of the two subpopulations "Left" and "Right" at a given point in time (averaged over a short time window). The color encodes time. In this example, the decision "Right" is made after about 900 milliseconds.


To get a better understanding of the network dynamics, we recommend to solve the exercise :doc:`spatial-working-memory`.

The parameters of our implementation differ from the original paper. In particular, the default network simulates only 480 spiking neurons which leads to relatively short simulation time even on less powerful computers.


**Book chapters**

Read the introduction of chapter `16, Competing populations and decision making  <http://neuronaldynamics.epfl.ch/online/Ch16.html>`_. To understand the mechanism of decision making in a network, read `16.2, Competition through common inhibition <http://neuronaldynamics.epfl.ch/online/Ch16.S2.html>`_.

If you have access to a scientific library, you may also want to read the original publication, :ref:`location-references` [1].

**Python classes**

The module :mod:`.competing_populations.decision_making` implements the network adapted from :ref:`location-references` [1, 2]. To get started, call the function  :func:`.competing_populations.decision_making.getting_started` or copy the following code into a Jupyter notebook.


.. code-block:: py

    %matplotlib inline
    from neurodynex.competing_populations import decision_making

    decision_making.getting_started()

Exercise: The network implementation
------------------------------------
Before we can analyse the decision making process and the simulation results, we first need to understand the structure of the network and how we can access the state variables of the respective subpopulations.

.. figure:: exc_images/DecisionMaking_NetworkStructureAll.png
    :align: center
    :width: 65%

    Network structure. The excitatory population is divided into three subpopulations, shown in the next figure.


.. figure:: exc_images/DecisionMaking_NetworkStructureDetail.png
    :align: center
    :width: 65%

    Structure within the excitatory population. The "Left" and "Right" subpopulations have strong recurrent weights :math:`(w^+ > w^0)` and weak projections to the other :math:`(w^- < w^0)`. All neurons receive a poisson input from an external source. Additionally, the neurons in the "Left" subpopulation receive poisson input with some rate :math:`\nu_{Left}`; the "Right" subpopulation receives a poisson input with a different rate :math:`\nu_{right}`.


Question: Understanding Brian2 Monitors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The network shown in the figure above is implemented in Brian2 in the function  :func:`.competing_populations.decision_making.sim_decision_making_network`. Each subpopulation is a `Brian2 NeuronGroup <http://brian2.readthedocs.io/en/stable/user/models.html>`_. Look at the source code of the function :func:`.sim_decision_making_network` to answer the following questions:


* For each of the four subpopulations, find the variable name of the corresponding `NeuronGroup <http://brian2.readthedocs.io/en/stable/user/models.html>`_.

* Each NeuronGroup is monitored with a `PopulationRateMonitor <http://brian2.readthedocs.io/en/stable/user/recording.html>`_, a `SpikeMonitor <http://brian2.readthedocs.io/en/stable/user/recording.html>`_, and a `StateMonitor <http://brian2.readthedocs.io/en/stable/user/recording.html>`_. Find the variable names for those monitors. Have a look at the `Brian2 documentation <http://brian2.readthedocs.io/en/stable/user/recording.html>`_ if you are not familiar with the concept of monitors.

* Which state variable of the neurons is recorded by the `StateMonitor <http://brian2.readthedocs.io/en/stable/user/recording.html>`_?


Question: Accessing a dictionary to plot the population rates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The monitors are returned in a `Python dictionary <https://docs.python.org/3/tutorial/datastructures.html?highlight=dictionary#dictionaries>`_ providing access to objects by name. Read the `Python documentation <https://docs.python.org/3/tutorial/datastructures.html?highlight=dictionary#dictionaries>`_ and look at the code block below or the function :func:`.competing_populations.decision_making.getting_started` to learn how dictionaries are used.

* Extend the following code block to include plots for all four subpopulations.
* Run the simulation for 800ms. What are the "typical" population rates of the four populations towards the end of the simulation? (In case the network did not decide, run the simulation again).
* Without running the simulation again, but by using the same ``results`` `dictionary <https://docs.python.org/3/tutorial/datastructures.html?highlight=dictionary#dictionaries>`_, plot the rates using different values for ``avg_window_width``.
* Interpret the effect of a very short and a very long averaging window.
* Find a value ``avg_window_width`` for which the population activity plot gives meaningful rates.


 .. code-block:: py

    import brian2 as b2
    from neurodynex.tools import plot_tools
    from neurodynex.competing_populations import decision_making
    import matplotlib.pyplot as plt

    results = decision_making.sim_decision_making_network(t_stimulus_start= 50. * b2.ms,
                                                          coherence_level=-0.6, max_sim_time=1000. * b2.ms)
    plot_tools.plot_network_activity(results["rate_monitor_A"], results["spike_monitor_A"],
                                     results["voltage_monitor_A"], t_min=0. * b2.ms, avg_window_width=2. * b2.ms,
                                     sup_title="Left")
    plot_tools.plot_network_activity(results["rate_monitor_B"], results["spike_monitor_B"],
                                     results["voltage_monitor_B"], t_min=0. * b2.ms, avg_window_width=2. * b2.ms,
                                     sup_title="Right")
    plt.show()


Remark: The parameter ``avg_window_width`` is passed to the function `PopulationRateMonitor.smooth_rate() <http://brian2.readthedocs.io/en/2.0.1/user/recording.html#recording-population-rates>`_ . This function is useful to solve one of the next exercises.

.. code-block:: py

    avg_window_width = 123*b2.ms
    sr = results["rate_monitor_A"].smooth_rate(window="flat", width=avg_window_width)/b2.Hz


Exercise: Stimulating the decision making circuit
-------------------------------------------------
The input stimulus is implemented by two inhomogenous Poisson processes: The subpopulation "Left" and "Right" receive input from two different PoissonGroups (see Figure "Network Structure"). The input has a ``coherence level c`` and is noisy. We have implemented this in the following way: every 30ms, the firing rates :math:`\nu_{left}` and :math:`\nu_{right}` of each of the two PoissonGroups are drawn from a normal distribution:


.. math::

   \nu_{left} &\sim& \mathcal{N}(\mu_{left},\,\sigma^{2})\\
   \nu_{right} &\sim& \mathcal{N}(\mu_{right},\,\sigma^{2})\\
   \mu_{left} &=& \mu_0 * (0.5 + 0.5c)\\
   \mu_{right} &=& \mu_0 * (0.5 - 0.5c)\\
   c &\in& [-1, +1]

The coherence level ``c``, the maximum mean :math:`\mu_0` and the standard deviation :math:`\sigma` are parameters of :func:`.sim_decision_making_network`.

Question: Coherence Level
~~~~~~~~~~~~~~~~~~~~~~~~~

* From the equation above, express the difference :math:`\mu_{left}-\mu_{right}` in terms of :math:`\mu_0` and :math:`c`.

* Find the distribution of the difference :math:`\nu_{left}-\nu_{right}`. Hint: the difference of two Gaussian distributions is another Gaussian distribution.

Now look at the documentation of the function :func:`.sim_decision_making_network` and find the default values of :math:`\mu_0` and :math:`\sigma`. Using those values, answer the following questions:

* What are the mean firing rates (in Hz) :math:`\mu_{left}` and :math:`\mu_{right}` for the coherence level c= -0.2?

* For c= -0.2, how does the difference :math:`\mu_{left}-\mu_{right}` compare to the variance of :math:`\nu_{left}-\nu_{right}`.


Question: Input stimuli with different coherence levels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run a few simulations with ``c=-0.1`` and ``c=+0.6``. Plot the network activity.

* Does the network always make the correct decision?
* Look at the population rates and estimate how long it takes the network to make a decision.


Exercise: Decision Space
------------------------

We can visualize the dynamics of the decision making process by plotting the activities of the two subpopulations "Left" / "Right" in a phase plane (see figure at the top of this page). Such a phase plane of competing states is also known as the *Decision Space*. A discussion of the decision making process in the decision space is out of the scope of this exercise but we refer to :ref:`location-references` [1].

Question: Plotting the Decision Space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Write a function that takes two `RateMonitors <http://brian2.readthedocs.io/en/2.0.1/user/recording.html#recording-population-rates>`_ and plots the *Decision Space*.

* Add a parameter ``avg_window_width`` to your function (same semantics as in the exercise above). Run a few simulations and plot the phase plane for different values of ``avg_window_width``.

* We can use a rate threshold as a decision criterion: We say the network has made a decision if one of the (smoothed) rates crosses a threshold. What are appropriate values for ``avg_window_width`` and ``rate threshold`` to detect a decision from the two rates?


Hint: Use Brian's smooth_rate function:

.. code-block:: py

    avg_window_width = 123*b2.ms
    sr = results["rate_monitor_A"].smooth_rate(window="flat", width=avg_window_width)/b2.Hz


Question: Implementing a decision criterion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Using your insights from the previous questions, implement a function **get_decision_time** that takes two `RateMonitors <http://brian2.readthedocs.io/en/2.0.1/user/recording.html#recording-population-rates>`_ , a ``avg_window_width`` and a ``rate_threshold``. The function should return a tuple (decision_time_left, decision_time_right). The decision time is the time index when some decision boundary is crossed. Possible return values are (1234.5ms, 0ms) for decision "Left", (0ms, 987.6ms) for decision "Right" and (0ms, 0ms) for the case when no decision is made within the simulation time. A return value like (123ms, 456ms) is an error and occurs if your function is called with inappropriate values for ``avg_window_width`` and ``rate_threshold``.

 The following code block shows how your function is called.

.. code-block:: py

    >> get_decision_time(results["rate_monitor_A"], results["rate_monitor_B"], avg_window_width=123*b2.ms, rate_threshold=45.6*b2.Hz)
    >> (0.543 * second, 0. * second)

The following code fragments could be useful:

.. code-block:: py

    smoothed_rates_A = rate_monitor_A.smooth_rate(window="flat", width=avg_window_width) / b2.Hz
    idx_A = numpy.argmax(smoothed_rates_A > rate_threshold/b2.Hz)
    t_A = idx_A * b2.defaultclock.dt

Run a few simulations to test your function.


Exercise: Percent-correct and Decision-time as a function of coherence level
----------------------------------------------------------------------------
We now investigate how the coherence level influences the decision making process. In order to estimate quantities like ``Percent-correct`` or ``Decision-time``, we have to average over multiple repetitions.

Question: Running multiple simulations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the function :func:`.competing_populations.decision_making.run_multiple_simulations` to get the values for multiple runs. Pass your function *get_decision_time* to :func:`.run_multiple_simulations` as shown here:

.. code-block:: py

    coherence_levels = [-0.1, -0.5]  # for negative values, B is the correct decision.
    nr_repetitions = 3

    time_to_A, time_to_B, count_A, count_B, count_No = decision_making.run_multiple_simulations(get_decision_time,coherence_levels, nr_repetitions, max_sim_time=??, rate_threshold=??, avg_window_width=??)

* See the doc of :func:`.run_multiple_simulations` to understand the parameters and return values.
* Write a function that takes ``coherence_levels, time_to_A, time_to_B, count_A, count_B, count_No`` and writes ``Percent correct`` (for each level in ``coherence_levels``) to the terminal.
* Think about other values you could get from the data. Add them to your function.


Question: Percent-Correct, Time-to-decision
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Using :func:`.run_multiple_simulations`, run at least 20 simulations for each of the two ``coherence_levels = [+0.15, -0.8]`` and visualize the results. Warning: Depending on your computer, this simulation could run for more than an hour.

* Visualize ``Percent correct`` versus ``coherence level``. Count simulations with "no decision" as wrong.

* Visualize ``Time to decision`` versus ``coherence level``. Ignore simulations with "no decision".

* Discuss your results.

* Optionally, if you have sufficient time/computing-power, you could run more levels.


.. code-block:: py

    import brian2 as b2
    from neurodynex.competing_populations import decision_making

    coherence_levels = [0.15, -0.8]
    nr_repetitions = 20

    # do not set other parameters (=defaults are used).
    time_to_A, time_to_B, count_A, count_B, count_No = decision_making.run_multiple_simulations(get_decision_time, coherence_levels, nr_repetitions, max_sim_time=1200 * b2.ms)

    # you may want to wrap the visualization into a function
    # plot_simulation_stats(coherence_levels, time_to_A, time_to_B, count_A, count_B, count_No)

.. _location-references:

**References**
--------------

[1] Wong, K.-F. & Wang, X.-J. A Recurrent Network Mechanism of Time Integration in Perceptual Decisions. J. Neurosci. 26, 1314–1328 (2006).

[2] Parts of this exercise and parts of the implementation are inspired by material from *Stanford University, BIOE 332: Large-Scale Neural Modeling, Kwabena Boahen & Tatiana Engel, 2013*, online available.
