"""
This file implements a type I and a type II model.
You can inject step currents and plot the responses.

Relevant book chapters:

- http://neuronaldynamics.epfl.ch/online/Ch4.S4.html

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
import pylab as plt
import numpy as np


def get_step_curr(I_tstart=20, I_tend=270, I_amp=.5):
    """Returns a pA step current TimedArray.

    Args:
        I_tstart (float): start of current step [ms]
        I_tend (float): start of end step [ms]
        I_amp (float): amplitude of current step [pA]

    Returns:
        StateMonitor: Brian2 StateMonitor with input current (I) and
        voltage (V) recorded
    """

    # 1ms sampled step current
    tmp = np.zeros(I_tend) * b2.uamp
    tmp[int(I_tstart):int(I_tend)] = I_amp * b2.pamp

    # This relies on the property of TimedArrays that
    # the returned value is equal to the final array
    # value of the passed array. Here, the step
    # will be 0 for all times > I_tend.
    return b2.TimedArray(tmp, dt=1.*b2.ms)


def plot_data(rec, title=None):
    """Plots a TimedArray for values I, v and w

    Args:
        rec (TimedArray): the data to plot
        title (string): plot title to display

    Returns:
        StateMonitor: Brian2 StateMonitor with input current (I) and
        voltage (V) recorded
    """
    # get data from rec
    t = rec.t / b2.ms
    v = rec.v[0] / b2.mV
    w = rec.w[0] / b2.mV
    I = rec.I[0] / b2.pA

    # plot voltage time series
    plt.subplot(311)
    plt.plot(t, v, lw=2)
    plt.xlabel('t [ms]')
    plt.ylabel('v [mV]')
    plt.grid()

    # plot activation and inactivation variables
    plt.subplot(312)
    plt.plot(t, w, 'k', lw=2)
    plt.xlabel('t [ms]')
    plt.ylabel('w [mV]')
    plt.grid()

    # plot current
    plt.subplot(313)
    plt.plot(t, I, lw=2)
    plt.axis((0, t.max(), 0, I.max()*1.1))
    plt.xlabel('t [ms]')
    plt.ylabel('I [pA]')
    plt.grid()

    if title is not None:
        plt.suptitle(title)

    plt.show()


class NeuronAbstract(object):
    """Abstract base class for both neuron types.

    This stores its own recorder and network, allowing
    each neuron to be run several times with changing
    currents while keeping the same neurogroup object
    and network internally.
    """

    def __init__(self):
        self.make_neuron()
        self.rec = b2.StateMonitor(self.neuron, ['v', 'w', 'I'], record=True)
        self.net = b2.Network([self.neuron, self.rec])
        self.net.store()

    def make_neuron(self, curr):
        """Abstract function, which creates neuron attribute for this class.

        Args:
            curr (TimedArray): Input current injected into the neuron
        """
        raise NotImplementedError

    def run(self, curr, simtime):
        """Runs the neuron for a given current.

        Args:
            curr (TimedArray): Input current injected into the neuron
            simtime (float): Simulation time [seconds]

        Returns:
            StateMonitor: Brian2 StateMonitor with input current (I) and
            voltage (V) recorded
        """

        self.net.restore()
        self.neuron.namespace["curr"] = curr

        # run the simulation
        self.net.run(simtime * b2.ms)

        return self.rec

    def step(self, tend, do_plot=True, I_tstart=20, I_tend=270, I_amp=.5):
        """Runs the neuron for a step current and plots the data.

        Args:
            tend (float): the simulation time of the model [ms]
            I_tstart (float): start of current step [ms]
            I_tend (float): start of end step [ms]
            I_amp (float): amplitude of current step [nA]
            do_plot (bool): plot the resulting simulation

        Returns:
            StateMonitor: Brian2 StateMonitor with input current (I) and
            voltage (V) recorded
        """

        curr = get_step_curr(I_tstart, I_tend, I_amp)
        rec = self.run(curr, tend)

        if do_plot:
            plot_data(
                rec,
                title="Step current",
            )

        return rec


class NeuronTypeOne(NeuronAbstract):

    def make_neuron(self):
        """Implements a type1 neuron model."""

        # forming the neuron model using differential equations
        eqs = '''
        I = curr(t) : amp
        dv/dt = (v - (v**3)/(3*mvolt*mvolt) - w + I*Gohm)/ms : volt
        dw/dt = (a*(v+0.7*mvolt)-w)/tau : volt
        '''

        self.neuron = b2.NeuronGroup(1, eqs)
        self.neuron.v = 0

        self.neuron.namespace['a'] = 1.25
        self.neuron.namespace['tau'] = 15.6 * b2.ms


class NeuronTypeTwo(NeuronAbstract):

    def make_neuron(self):
        """Implements a type2 neuron model."""

        # neuron parameters
        pars = {
            "g_1": 4.4 * (1/b2.mV),
            "g_2": 8 * (1/b2.mV),
            "g_L": 2,
            "V_1": 120 * b2.mV,
            "V_2": -84 * b2.mV,
            "V_L": -60 * b2.mV,
            "phi": 0.06666667,
            "R": 100 * b2.Gohm,
        }

        # forming the neuron model using differential equations
        eqs = '''
        I = curr(t) : amp
        winf = (0.5*mV)*( 1 + tanh((v-12*mV)/(17*mV)) ) : volt
        tau = (1*ms)/cosh((v-12*mV)/(2*17*mV)) : second
        m = (0.5*mV)*(1+tanh((v+1.2*mV)/(18*mV))) : volt
        dv/dt = (-g_1*m*(v-V_1) - g_2*w*(v-V_2) - g_L*(v-V_L) \
            + I*R)/(20*ms) : volt
        dw/dt = phi*(winf-w)/tau : volt
        '''

        self.neuron = b2.NeuronGroup(1, eqs)
        self.neuron.v = pars["V_L"]
        self.neuron.namespace.update(pars)
