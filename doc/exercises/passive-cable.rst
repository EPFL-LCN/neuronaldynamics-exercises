Dendrites and the (passive) cable equation
==========================================

**Book chapters**

In `Chapter 3 Section 2 <Chapter_>`_ the cable equation is derived and compartmental models are introduced.

.. _Chapter: http://neuronaldynamics.epfl.ch/online/Ch3.S2.html

**Python classes**

The :mod:`.cable_equation.passive_cable` module implements a passive cable using a `Brian2 multicompartment <http://brian2.readthedocs.io/en/latest/user/multicompartmental.html>`_ model. To get started, import the module and call the demo function:

.. code-block:: py

    import brian2 as b2
    import matplotlib.pyplot as plt
    from neurodynex.cable_equation import passive_cable
    from neurodynex.tools import input_factory
    passive_cable.getting_started()

The function :func:`.passive_cable.getting_started` injects a very short pulse current at (t=500ms, x=100um) into a finite length cable and then lets Brian evolve the dynamics for 2ms. This simulation produces a time x location matrix whose entries are the membrane voltage at each (time,space)-index. The result is visualized using pyplot.imshow.


.. figure:: exc_images/cable_equation_pulse.png
   :align: center


.. note::

    The axes in the figure above are not scaled to the physical units but show the raw matrix indices. These indices depend on the spatial resolution (number of compartments) and the temporal resolution (brian2.defaultclock.dt). For the exercises make sure you correctly scale the units using Brian's `unit system <http://brian2.readthedocs.io/en/latest/user/units.html>`_ . As an example, to plot voltage vs. time you call

    .. code::

        pyplot.plot(voltage_monitor.t / b2.ms, voltage_monitor[0].v / b2.mV)

    This way, your plot shows voltage in mV and time in ms, which is useful for visualizations. Note that this scaling (to physical units) is different from the scaling in the theoretical derivation (e.g. `chapter 3.2.1 <Chapter_>`_  where the quantities are rescaled to a unit-free characteristic length scale


Using the module :mod:`.cable_equation.passive_cable`, we study some properties of the passive cable. Note: if you do not specify the cable parameters, the function :func:`.cable_equation.passive_cable.simulate_passive_cable` uses the following default values:

.. code-block:: py

    CABLE_LENGTH = 500. * b2.um  # length of dendrite
    CABLE_DIAMETER = 2. * b2.um  # diameter of dendrite
    R_LONGITUDINAL = 0.5 * b2.kohm * b2.mm  # Intracellular medium resistance
    R_TRANSVERSAL = 1.25 * b2.Mohm * b2.mm ** 2  # cell membrane resistance (->leak current)
    E_LEAK = -70. * b2.mV  # reversal potential of the leak current (-> resting potential)
    CAPACITANCE = 0.8 * b2.uF / b2.cm ** 2  # membrane capacitance

You can easily access those values in your code:

.. code-block:: py

    from neurodynex.cable_equation import passive_cable
    print(passive_cable.R_TRANSVERSAL)



Exercise: spatial and temporal evolution of a pulse input
---------------------------------------------------------
Create a cable of length 800um and inject a 0.1ms long step current of amplitude 0.8nanoAmp at (t=1ms, x=200um). Run Brian for 3 milliseconds.

You can use the function :func:`.cable_equation.passive_cable.simulate_passive_cable` to implement this task. For the parameters not specified here (e.g. dentrite diameter) you can rely on the default values. Have a look at the documentation of :func:`.simulate_passive_cable` and the source code of :func:`.passive_cable.getting_started` to learn how to efficiently solve this exercise.
From the specification of :func:`.simulate_passive_cable` you should also note, that it returns two objects which are helpful to access the values of interest using spatial indexing:

.. code-block:: py

    voltage_monitor, cable_model = passive_cable.simulate_passive_cable(...)
    probe_location = 0.123 * b2.mm
    v = voltage_monitor[cable_model.morphology[probe_location]].v

Question:
~~~~~~~~~
#. What is the maximum depolarization you observe? Where and when does it occur?

#. Plot the temporal evolution (t in [0ms, 3ms]) of the membrane voltage at the locations 0um, 100um, ... , 600 um in one figure.

#. Plot the spatial evolution (x in [0um, 800um]) of the membrane voltage at the time points 1.0ms, 1.1ms, ... , 1.6ms in one plot

#. Discuss the figures.


Exercise: Spatio-temporal input pattern
---------------------------------------
While the passive cable use here is a very simplified model of a real dendrite, we can still get an idea of how input spikes would look to the soma. Imagine a dendrite of some length and the soma at x=0um. What is the depolarization at x=0 if the dendrite receives multiple spikes at different time/space locations? This is what we study in this exercise:

Create a cable of length 800uM and inject three short pulses A, B, and C at different time/space locations:
 | A: (t=1.0ms, x=100um)
 | B: (t=1.5ms, x=200um)
 | C: (t=2.0ms, x=300um)
 | Pulse input: 100us duration, 0.8nanoAmp amplitude

Make use of the function :func:`.input_factory.get_spikes_current` to easily create such an input pattern:

.. code-block:: py

    t_spikes = [10, 15, 20]
    l_spikes = [100. * b2.um, 200. * b2.um, 300. * b2.um]
    current = input_factory.get_spikes_current(t_spikes, 100*b2.us, 0.8*b2.namp, append_zero=True)
    voltage_monitor_ABC, cable_model = passive_cable.simulate_passive_cable(..., current_injection_location=l_spikes, input_current=current, ...)

Run Brian for 5 milliseconds. Your simulation for this input pattern should look similar to this figure:


.. figure:: exc_images/passive_cable_ABC.png
   :align: center
   :width: 60%


Question
~~~~~~~~
#. plot the temporal evolution (t in [0ms, 5ms]) of the membrane voltage at the soma (x=0). What is the maximal depolarization?
#. reverse the order of the three input spikes:

 | C: (t=1.0ms, x=300um)
 | B: (t=1.5ms, x=200um)
 | A: (t=2.0ms, x=100um)

Again, let Brian simulate 5 milliseconds. In the same figure as before, plot the temporal evolution (t in [0ms, 5ms]) of the membrane voltage at the soma (x=0). What is the maximal depolarization? Discuss the result.


Exercise: Effect of cable parameters
------------------------------------
So far, you have called the function :func:`.simulate_passive_cable` without specifying the cable parameters. That means, the model was run with the default values. Look at the documentation of :func:`.simulate_passive_cable` to see which parameters you can change.

Keep in mind that our cable model is very simple compared to what happens in  dendrites or axons. But we can still observe the impact of a parameter change on the current flow. As an example, think of a myelinated fiber: it has a much **lower** membrane capacitance and **higher** membrane resistance. Let's compare these two parameter-sets:

Question
~~~~~~~~
Inject a very brief pulse current at (t=.05ms, x=400um). Run Brian twice for 0.2 ms with two different parameter sets (see example below). Plot the temporal evolution of the membrane voltage at x=500um for the two parameter sets. Discuss your observations.


 Note: to better see some of the effects, plot only a short time window and increase the temporal resolution of the numerical approximation (b2.defaultclock.dt = 0.005 * b2.ms)

.. code-block:: py

    # set 1: (same as defaults)
    membrane_resistance_1 = 1.25 * b2.Mohm * b2.mm ** 2
    membrane_capacitance_1 = 0.8 * b2.uF / b2.cm ** 2
    # set 2: (you can think of a myelinated "cable")
    membrane_resistance_2 = 5.0 * b2.Mohm * b2.mm ** 2
    membrane_capacitance_2 = 0.2 * b2.uF / b2.cm ** 2



Exercise: stationary solution and comparison with theoretical result
--------------------------------------------------------------------
Create a cable of length 500um and inject a **constant current** of amplitude 0.1nanoAmp at x=0um. You can use the input_factory to create that current. Note the parameter append_zero=False. As we are not interested in the exact values of the transients, we can speed up the simulation increase the width of a timestep dt: b2.defaultclock.dt = 0.1 * b2.ms

.. code-block:: py

        b2.defaultclock.dt = 0.1 * b2.ms
        current = input_factory.get_step_current(0, 0, unit_time=b2.ms, amplitude=0.1 * b2.namp, append_zero=False)
        voltage_monitor, cable_model = passive_cable.simulate_passive_cable(
        length=0.5 * b2.mm, current_injection_location = [0*b2.um],
        input_current=current, simulation_time=sim_time, nr_compartments=N_comp)
        v_X0 = voltage_monitor.v[0,:]  # access the first compartment
        v_Xend = voltage_monitor.v[-1,:]  # access the last compartment
        v_Tend = voltage_monitor.v[:, -1]  # access the last time step

Question
~~~~~~~~
Before running a simulation, sketch two curves, one for x=0um and one for x=500um, of the membrane potential Vm versus time. What steady state Vm do you expect?

Now run the Brian simulator for 100 milliseconds.

#. Plot Vm vs. time (t in [0ms, 100ms]) at x=0um and x=500um and compare the curves to your sketch.
#. Plot Vm vs location (x in [0um, 500um]) at t=100ms.


Question
~~~~~~~~
#. Compute the characteristic length :math:`\lambda` (= length scale = lenght constant) of the cable. Compare your value with the previous figure.

 :math:`\lambda=\sqrt{\frac{r_{Membrane}}{r_{Longitudinal}}}`

Question (Bonus)
~~~~~~~~~~~~~~~~
You observed that the membrane voltage reaches a location dependent steady-state value. Here we compare those simulation results to the analytical solution.


#. Derive the analytical steady-state solution. (finite cable length L, constant current I0 at x=0, sealed end: no longitudinal current at x=L).

#. Plot the analytical solution and the simulation result in one figure.

#. Run the simulation with different resolution parameters (change defaultclock.dt and/or the number of compartments). Compare the simulation with the analytical solution.

#. If you need help to get started, or if you're not sure about the analytical solution, you can find a solution in the `Brian2 docs <http://brian2.readthedocs.io/en/latest/examples/compartmental.cylinder.html>`_ :