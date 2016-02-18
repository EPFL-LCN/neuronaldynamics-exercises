"""
This file implements a leaky intergrate-and-fire (LIF) model.
You can inject a step current or sinusoidal current into
neuron using LIF_Step() or LIF_Sinus() methods respectively.

For associated exercises, see `exercise.pdf` in the same folder.

Relevant book chapters:

- http://neuronaldynamics.epfl.ch/online/Ch1.S3.html

"""

# This file is part of the exercise code repository accompanying
# the book: Neuronal Dynamics (see http://neuronaldynamics.epfl.ch)
# located at http://github.com/EPFL-LCN/neuronaldynamics-exercises.

# This free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License 2.0 as published by the
# Free Software Foundation. You should have received a copy of the
# GNU General Public License along with the repository. If not,
# see http://www.gnu.org/licenses/.

# Should you reuse and publish the code for your own purposes,
# please cite the book or point to the webpage http://neuronaldynamics.epfl.ch.

# Wulfram Gerstner, Werner M. Kistler, Richard Naud, and Liam Paninski.
# Neuronal Dynamics: From Single Neurons to Networks and Models of Cognition.
# Cambridge University Press, 2014.

import brian2 as b2
from brian2.units.stdunits import nA
import pylab as plt
import numpy as np

# constants
v_reset = 0.*b2.mV
v_threshold = 1.*b2.mV
R = 1*b2.Mohm
v_rest = 0*b2.mV
tau = 1*b2.ms


def do_plot(rec, v_threshold=None, title=None):

    """Plots a TimedArray for values I and v

    Args:
        rec (TimedArray): the data to plot
        v_threshold (float): plots a threshold at this level [mV]
        title (string): plot title to display
    """

    plt.subplot(211)
    plt.plot(rec.t/b2.ms, rec.v[0]/b2.mV, lw=2)

    if v_threshold is not None:
        plt.plot(
            (rec.t/b2.ms)[[0, -1]],
            [v_threshold/b2.mV, v_threshold/b2.mV],
            'r--', lw=2
        )

    plt.xlabel('t [ms]')
    plt.ylabel('v [mV]')
    plt.ylim(0, v_threshold/b2.mV * 1.2)
    plt.grid()

    plt.subplot(212)
    plt.plot(rec.t/b2.ms, rec.I[0]/nA, lw=2)
    plt.xlabel('t [ms]')
    plt.ylabel('I [mV]')
    plt.grid()

    if title is not None:
        plt.suptitle(title)

    plt.show()


def LIF_Neuron(curr, simtime):

    """Simple LIF neuron implemented in Brian2.

    Args:
        curr (TimedArray): Input current injected into the neuron
        simtime (float): Simulation time [seconds]

    Returns:
        StateMonitor:   Brian2 StateMonitor with input current (I) and
                        voltage (V) recorded
    """

    v_reset_ = "v=%f*volt" % v_reset
    v_threshold_ = "v>%f*volt" % v_threshold

    # differential equation of Leaky Integrate-and-Fire model
    eqs = '''
        dv/dt = ( -(v-v_rest) + R * I ) / tau : volt
        I = curr(t) : amp
    '''

    # LIF neuron using Brian2 library
    IF = b2.NeuronGroup(1, model=eqs, reset=v_reset_, threshold=v_threshold_)
    IF.v = v_rest

    # monitoring membrane potential of neuron and injecting current
    rec = b2.StateMonitor(IF, ['v', 'I'], record=True)

    # run the simulation
    b2.run(simtime)

    return rec


def LIF_Step(I_tstart=20, I_tend=70, I_amp=1.005, tend=100):

    """Run the LIF and give a step current input.

    Args:
        tend (float): the simulation time of the model [ms]
        I_tstart (float): start of current step [ms]
        I_tend (float): start of end step [ms]
        I_amp (float): amplitude of current step [nA]

    """

    # 1ms sampled step current
    tmp = np.zeros(tend) * nA
    tmp[int(I_tstart):int(I_tend)] = I_amp * nA
    curr = b2.TimedArray(tmp, dt=1.*b2.ms)

    do_plot(
        LIF_Neuron(curr, tend * b2.ms),
        title="Step current",
        v_threshold=v_threshold
    )

    return True


def LIF_Sinus(I_freq=0.1, I_offset=0.5, I_amp=0.5, tend=100, dt=.1):

    """
    Run the LIF for a sinusoidal current

    Args:
        tend (float): the simulation time of the model [ms]
        I_freq (float): frequency of current sinusoidal [kHz]
        I_offset (float): DC offset of current [nA]
        I_amp (float): amplitude of sinusoidal [nA]

    """

    # dt sampled sinusoidal function
    t = np.arange(0, tend, dt)
    tmp = (I_amp*np.sin(2.0*np.pi*I_freq*t)+I_offset) * nA
    curr = b2.TimedArray(tmp, dt=dt*b2.ms)

    do_plot(
        LIF_Neuron(curr, tend * b2.ms),
        title="Sinusoidal current",
        v_threshold=v_threshold
    )

    return True
