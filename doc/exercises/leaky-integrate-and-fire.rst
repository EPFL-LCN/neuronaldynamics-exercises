Leaky-integrate-and-fire model
==============================

**Book chapters**

See `Chapter 1 Section 3 <Chapter_>`_ on general information about
leaky-integrate-and-fire models.

.. _Chapter: http://neuronaldynamics.epfl.ch/online/Ch1.S3.html


**Python classes**

The :mod:`.leaky_integrate_and_fire.LIF` module contains all code required for this exercise.
At the beginning of your exercise solutions, import the contained functions by running

.. code-block:: py

	from neurodynex.leaky_integrate_and_fire.LIF import *

You can then simply run the exercise functions by executing

.. code-block:: py

	LIF_Step()  # example Step
	LIF_Sinus()  # example Sinus

Exercise
--------

Use the function :func:`.LIF_Step` to simulate a Leaky Integrate-And-Fire
neuron stimulated by a current step of a given amplitude. The goal of
this exercise is to modify the provided python functions and use the
``numpy`` and ``matplotlib`` packages to answer the following questions.

Question
~~~~~~~~

What is the minimum current step amplitude ``I_amp`` to elicit a spike
with model parameters as given in :func:`.LIF_Step`? Plot the injected
values of current step amplitude against the frequency of the spiking
response (you can use the inter-spike interval to calculate this – let
the frequency be :math:`0Hz` if the model does not spike, or emits only
a single spike) during a :math:`500ms` current step.

Exercise
--------

Use the function :func:`.LIF_Sinus` to simulate a Leaky Integrate-And-Fire
neuron stimulated by a sinusoidal current of a given frequency. The goal
of this exercise is to modify the provided python functions and use the
``numpy`` and ``matplotlib`` packages to plot the amplitude and frequency
gain and phase of the voltage oscillations as a function of the input
current frequency.

Question
~~~~~~~~

For input frequencies between :math:`0.1kHz` and :math:`1.kHz`, plot the
input frequency against the resulting *amplitude of subthreshold
oscillations* of the membrane potential. If your neuron emits spikes at
high stimulation frequencies, decrease the amplitude of the input
current.

Question
~~~~~~~~

For input frequencies between :math:`0.1kHz` and :math:`1.kHz`, plot the
input frequency against the resulting *frequency and phase of
subthreshold oscillations* of the membrane potential. Again, keep your
input amplitude in a regime, where the neuron does not fire action
potentials.