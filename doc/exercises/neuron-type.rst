Type I and type II neuron models
================================

**Book chapters**

See `Chapter 4 <Chapter4_>`_ and especially `Chapter 4 Section 4 <Chapter44_>`_ for background knowledge on Type I and Type II neuron
models.

.. _Chapter4: http://neuronaldynamics.epfl.ch/online/Ch4.html
.. _Chapter44: http://neuronaldynamics.epfl.ch/online/Ch4.S4.html

**Python classes**

The :mod:`neurodynex.neuron_type.neurons` module contains all classes required for this exercise. To get started, call :func:`getting_started <.neuron_type.neurons.getting_started>` or copy the following code into your Jupyter notebook:

.. code-block:: python

    %matplotlib inline  # needed in Notebooks, not in Python scripts
    import brian2 as b2
    import matplotlib.pyplot as plt
    import numpy as np
    from neurodynex.tools import input_factory, plot_tools, spike_tools
    from neurodynex.neuron_type import neurons

    # create an input current
    input_current = input_factory.get_step_current(50, 150, 1.*b2.ms, 0.5*b2.pA)

    # get one instance of class NeuronX and save that object in the variable 'a_neuron_of_type_X'
    a_neuron_of_type_X = neurons.NeuronX()  # we do not know if it's type I or II
    # simulate it and get the state variables
    state_monitor = a_neuron_of_type_X.run(input_current, 200*b2.ms)
    # plot state vs. time
    neurons.plot_data(state_monitor, title="Neuron of Type X")

    # get an instance of class NeuronY
    a_neuron_of_type_Y = neurons.NeuronY()  # we do not know if it's type I or II
    state_monitor = a_neuron_of_type_Y.run(input_current, 200*b2.ms)
    neurons.plot_data(state_monitor, title="Neuron of Type Y")


.. note::

    For those who are interested, `here is more about classes and inheritance in Python <https://en.wikibooks.org/wiki/Python_Programming/Classes>`_.

Exercise: Probing Type I and Type II neuron models
--------------------------------------------------

This exercise deals not only with Python functions, but with python objects. The classes :class:`NeuronX <.neuron_type.neurons.NeuronX>` and :class:`NeuronY <.neuron_type.neurons.NeuronY>` both are neurons, that have different dynamics: **one is Type I and one is Type II**. Finding out which class implements which dynamics is the goal of the exercise.


The types get randomly assigned each time you load the module or you call the function :func:`.neurons.neurontype_random_reassignment`.

Question: Estimating the threshold
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

What is the threshold current for repetitive firing for :class:`NeuronX <.neuron_type.neurons.NeuronX>` and :class:`NeuronY <.neuron_type.neurons.NeuronY>`?

Exploring various values of ``I_amp``, find the range in which the threshold occurs, to a precision of 0.01.

Plot the responses to step current which starts after 100ms (to let the system equilibrate) and lasting at least 1000ms (to detect repetitive firing with a long period). You can do this by modifying the code example given above. Make sure to check the documentation of the functions you use: :func:`.input_factory.get_step_current`, :func:`.neuron_type.neurons.run` and :func:`.neuron_type.neurons.plot_data`.

Already from the voltage response near threshold you might have an idea which is type I or II, but let’s investigate further.


Exercise: f-I curves
--------------------

In this exercise you will write a python script that plots the f-I curve for type I and type II neuron models.

Get firing rates from simulations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We provide you with a function :func:`.spike_tools.get_spike_time` to determine the spike times from a StateMonitor. The following code shows how to use that function. Note that the return value is a Brian Quantity: it has units. If you write code using units, you'll get consistency checks done by Brian.

.. code-block:: python

    input_current = input_factory.get_step_current(100, 110, b2.ms, 0.5*b2.pA)
    state_monitor = a_neuron_of_type_X.run(input_current, ...)
    spike_times = spike_tools.get_spike_time(state_monitor, ...)
    print(spike_times)
    print(type(spike_times))  # it's a Quantity

Now **write a new function** (in your own `.py` file or in your Jupyter Notebook) that calculates an estimate of the firing rate. In your function use :func:`.spike_tools.get_spike_time`

.. code-block:: python

    def get_firing_rate(neuron, input_current, spike_threshold):

        # inject a test current into the neuron and call it's run() function.
        # get the spike times using spike_tools.get_spike_times
        # from the spike times, calculate the firing rate f

        return f

.. note::

    To calculate the firing rate, first calculate the inter-spike intervals (time difference between spikes) from the spike times using this elegant indexing idiom

    .. code-block:: python

        isi = st[1:]-st[:-1]

    Then find the mean isi and take the reciprocal to yield the firing-rate. As these are standard operations, you can expect that someone else has already implemented it. Have a look at the numpy package and look up the functions diff and mean. Once you have implemented your function, you should verify it's correctness: inject a few currents into your neuron, plot the voltage response and compare the plot with the firing rate computed by your function.


.. note::

    You can check your results by calling:
        
    .. code-block:: python
        
        spike_tools.pretty_print_spike_train_stats(...)


Plot the f-I curve
~~~~~~~~~~~~~~~~~~

Now let’s use your function ``get_firing_rate`` to plot an f-vs-I curve for both neuron classes.

Add the following function skeleton to your code and complete it to plot the f-I curve, given the neuron class as an argument:

.. code-block:: python

    import matplotlib.pyplot as plt
    import numpy as np

    def plot_fI_curve(NeuronClass):
        
        plt.figure()  # new figure
        
        neuron = NeuronClass()  # instantiate the neuron class

        I = np.arange(0.0,1.1,0.1)  # a range of current inputs
        f = []

        # loop over current values
        for I_amp in I:
            
            firing_rate = # insert here a call to your function get_firing_rate( ... )

            f.append(firing_rate)
    
        plt.plot(I, f)
        plt.xlabel('Amplitude of Injecting step current (pA)')
        plt.ylabel('Firing rate (Hz)')
        plt.grid()
        plt.show()


* Call your ``plot_fI_curve`` function with each class ``NeuronX`` and ``NeuronY`` as argument.
* Change the ``I`` range (and reduce the step size) to zoom in near the threshold, and try running it again for both classes.

Which class is Type I and which is Type II? Check your result:

.. code-block:: py

    print("a_neuron_of_type_X is : {}".format(a_neuron_of_type_X.get_neuron_type()))
    print("a_neuron_of_type_Y is : {}".format(a_neuron_of_type_Y.get_neuron_type()))