Type I and type II neuron models
================================

**Book chapters**

See `Chapter 4 <Chapter4_>`_ and especially `Chapter 4 Section 4 <Chapter44_>`_ for background knowledge on Type I and Type II neuron
models.

.. _Chapter4: http://neuronaldynamics.epfl.ch/online/Ch4.html
.. _Chapter44: http://neuronaldynamics.epfl.ch/online/Ch4.S4.html

**Python classes**

The :mod:`neurodynex.neuron_type.typeXY` module contains all classes required for this exercise.
For the exercises you will need to import the classes :class:`NeuronX <.neuron_type.typeXY.NeuronX>` and :class:`NeuronY <.neuron_type.typeXY.NeuronY>` by running

.. code-block:: python

    from neurodynex.neuron_type.typeXY import NeuronX, NeuronY

.. note::

    Both :class:`NeuronX <.neuron_type.typeXY.NeuronX>` and :class:`NeuronY <.neuron_type.typeXY.NeuronY>` inherit from a common base class :class:`.neuron_type.neurons.NeuronAbstract` and thus implement similar methods.

    For those who are interested, `here is more about classes and inheritance in Python <https://en.wikibooks.org/wiki/Python_Programming/Classes>`_.

Exercise: Probing Type I and Type II neuron models
--------------------------------------------------

This exercise deals not only with Python functions, but with python objects.

The classes :class:`NeuronX <.neuron_type.typeXY.NeuronX>` and :class:`NeuronY <.neuron_type.typeXY.NeuronY>` both are neurons, that have different dynamics: **one is Type I and one is Type II**. Finding out which class implements which dynamics is the goal of the exercise.

To run the exercises you will have to instantiate these classes. You can then plot step_current injections (using the :meth:`step <.neuron_type.neurons.NeuronAbstract.step>` method) or extract the firing rate for a given step current (using the :meth:`get_rate <.neuron_type.neurons.NeuronAbstract.get_rate>` method):

.. code-block:: python

    from neurodynex.neuron_type.typeXY import NeuronX, NeuronY

    n1 = NeuronX()  # instantiates a new neuron of type X

    n1.step(do_plot=True)  # plot a step current injection
    

To check your results, you can use the :meth:`get_neuron_type <.neuron_type.typeXY.NeuronX.get_neuron_type>` function, e.g.:

.. code-block:: python

    >> n1 = NeuronX()  # instantiates a new neuron of type X
    >> n1.get_neuron_type()
    neurodynex.neuron_type.neurons.NeuronTypeOne

Question: Estimating the threshold
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

What is the threshold current for repetitive firing for :class:`NeuronX <.neuron_type.typeXY.NeuronX>` and :class:`NeuronY <.neuron_type.typeXY.NeuronY>`?

Exploring various values of ``I_amp``, find the range in which the
threshold occurs, to a precision of 0.01.

.. note::

    As shown abve, use the :meth:`step <.neuron_type.neurons.NeuronAbstract.step>` functions to plot the responses to step current which starts after 100ms (to let the system equilibrate) and lasting at least 1000ms (to detect repetitive firing with a long period):

Already from the voltage response near threshold you might have an idea which is type I or II, but let’s investigate further.

Question: Pulse response
~~~~~~~~~~~~~~~~~~~~~~~~

Plot the response to short current pulses near threshold, and
interpret the results: which class is Type I, which is II?

For example:

.. code-block:: python

    import matplotlib.pyplot as plt
    plt.figure()  # new figure
    n1 = NeuronX()  # instantiates a new neuron of type X
    
    t, v, w, I = n1.step(I_amp=1.05, I_tstart=100, I_tend=110, t_end=300)
    plt.plot(t,v)

    t, v, w, I = n1.step(I_amp=1.1, I_tstart=100, I_tend=110, t_end=300)
    plt.plot(t,v)

    # can you simplify this in a loop?

    plt.show()

Exercise: f-I curves
--------------------

During the questions of this exercise you will write a python script that plots the f-I curve for type I and type II neuron models.

Get firing rates from simulations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We provide you with a function :func:`get_spiketimes <.neuron_type.neurons.get_spiketimes>` to determine the spike times from
given timeseries ``t`` and ``v``:

.. code-block:: python
    
    >> from neurodynex.neuron_type.neurons import get_spiketimes
    >> t, v, w, I = n1.step(I_amp=1.0, I_tstart=100, I_tend=1000., t_end=1000.)
    >> st = get_spiketimes(t, v)
    >> print st
    [ 102.9  146.1   189.1 ... ]

Use this function to write a Python function (in your own `.py` file) that calculates an estimate of the firing rate, given a neuron instance and an input current:

.. code-block:: python

    def get_firing_rate(neuron, I_amp):

        # run a step on the neuron via neuron.step()
        # get the spike times
        # calculate the firing rate f

        return f

.. note::

    To calculate the firing rate, first calculate the inter-spike intervals (time difference between spikes) from the spike times using this elegant indexing idiom

    .. code-block:: python

        isi = st[1:]-st[:-1]

    Then find the mean and take the reciprocal (pay attention when
    converting from 1/ms to Hz) to yield the firing-rate:

    .. code-block:: python

        f = 1000.0/mean(isi)

.. note::

    You can check your results by calling:
        
    .. code-block:: python
        
        # get firing rate and plot the dynamics for an injection of I_amp
        n1.get_rate(I_amp, do_plot=True)


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

        I = np.arange(0.0,1.05,0.1)  # a range of current inputs
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
* Change the ``I`` range to zoom in near the threshold, and try running it again for both classes.

Which class is Type I and which is Type II?
