"""
This file implements a type I and a type II model from
the abstract base class NeuronAbstract.

You can inject step currents and plot the responses,
as well as get firing rates.

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
import matplotlib.pyplot as plt
import numpy as np
import random
import sys


def plot_data(state_monitor, title=None, show=True):
    """Plots a TimedArray for values I, v and w

    Args:
        state_monitor (StateMonitor): the data to plot. expects ["v", "w", "I"] and (by default) "t"
        title (string, optional): plot title to display
        show (bool, optional): call plt.show for the plot

    Returns:
        StateMonitor: Brian2 StateMonitor with input current (I) and
            voltage (V) recorded
    """

    t = state_monitor.t / b2.ms
    v = state_monitor.v[0] / b2.mV
    w = state_monitor.w[0] / b2.mV
    I_ext = state_monitor.I[0] / b2.pA

    # plot voltage time series
    plt.figure()
    plt.subplot(311)
    plt.plot(t, v, lw=2)
    plt.xlabel("t [ms]")
    plt.ylabel("v [mV]")
    plt.grid()

    # plot activation and inactivation variables
    plt.subplot(312)
    plt.plot(t, w, "k", lw=2)
    plt.xlabel("t [ms]")
    plt.ylabel("w [mV]")
    plt.grid()

    # plot current
    plt.subplot(313)
    plt.plot(t, I_ext, lw=2)
    plt.axis((0, t.max(), 0, I.max() * 1.1))
    plt.xlabel("t [ms]")
    plt.ylabel("I [pA]")
    plt.grid()

    if title is not None:
        plt.suptitle(title)

    if show:
        plt.show()


class NeuronAbstract(object):
    """Abstract base class for both neuron types.

    This stores its own recorder and network, allowing
    each neuron to be run several times with changing
    currents while keeping the same neurogroup object
    and network internally.
    """

    def __init__(self):
        self._make_neuron()
        self.rec = b2.StateMonitor(self.neuron, ["v", "w", "I"], record=True)
        self.net = b2.Network([self.neuron, self.rec])
        self.net.store()

    def _make_neuron(self):
        """Abstract function, which creates neuron attribute for this class."""

        raise NotImplementedError

    def get_neuron_type(self):
        """
        Type I or II.

        Returns:
            type as a string "Type I" or "Type II"
        """
        return self._get_neuron_type()

    def _get_neuron_type(self):
        """Just a trick to have the underlying function NOT being documented by sphinx
        (because this function's name starts with _)"""
        raise NotImplementedError

    def run(self, input_current, simtime):
        """Runs the neuron for a given current.

        Args:
            input_current (TimedArray): Input current injected into the neuron
            simtime (Quantity): Simulation time in correct Brian units.

        Returns:
            StateMonitor: Brian2 StateMonitor with input current (I) and
            voltage (V) recorded
        """

        self.net.restore()
        self.neuron.namespace["input_current"] = input_current

        # run the simulation
        self.net.run(simtime)

        return self.rec


class _NeuronTypeOne(NeuronAbstract):

    def _get_neuron_type(self):
        return "Type I"

    def _make_neuron(self):
        """Sets the self.neuron attribute."""

        # neuron parameters
        pars = {
            "g_1": 4.4 * (1 / b2.mV),
            "g_2": 8 * (1 / b2.mV),
            "g_L": 2,
            "V_1": 120 * b2.mV,
            "V_2": -84 * b2.mV,
            "V_L": -60 * b2.mV,
            "phi": 0.06666667,
            "R": 100 * b2.Gohm,
        }

        # forming the neuron model using differential equations
        eqs = """
        I = input_current(t,i) : amp
        winf = (0.5*mV)*( 1 + tanh((v-12*mV)/(17*mV)) ) : volt
        tau = (1*ms)/cosh((v-12*mV)/(2*17*mV)) : second
        m = (0.5*mV)*(1+tanh((v+1.2*mV)/(18*mV))) : volt
        dv/dt = (-g_1*m*(v-V_1) - g_2*w*(v-V_2) - g_L*(v-V_L) \
            + I*R)/(20*ms) : volt
        dw/dt = phi*(winf-w)/tau : volt
        """

        self.neuron = b2.NeuronGroup(1, eqs, method="euler")
        self.neuron.v = pars["V_L"]
        self.neuron.namespace.update(pars)


class _NeuronTypeTwo(NeuronAbstract):

    def _get_neuron_type(self):
        return "Type II"

    def _make_neuron(self):
        """Sets the self.neuron attribute."""

        # forming the neuron model using differential equations
        eqs = """
        I = input_current(t,i) : amp
        dv/dt = (v - (v**3)/(3*mvolt*mvolt) - w + I*Gohm)/ms : volt
        dw/dt = (a*(v+0.7*mvolt)-w)/tau : volt
        """

        self.neuron = b2.NeuronGroup(1, eqs, method="euler")
        self.neuron.v = 0

        self.neuron.namespace["a"] = 1.25
        self.neuron.namespace["tau"] = 15.6 * b2.ms


def neurontype_random_reassignment():
    """
    Randomly reassign the two types:
    Returns:

    """
    if random.random() < .5:
        NeuronX = type('NeuronX', _NeuronTypeOne.__bases__, dict(_NeuronTypeOne.__dict__))
        NeuronY = type('NeuronY', _NeuronTypeTwo.__bases__, dict(_NeuronTypeTwo.__dict__))
    else:
        NeuronX = type('NeuronX', _NeuronTypeTwo.__bases__, dict(_NeuronTypeTwo.__dict__))
        NeuronY = type('NeuronY', _NeuronTypeOne.__bases__, dict(_NeuronTypeOne.__dict__))
    thismodule = sys.modules[__name__]
    setattr(thismodule, "NeuronX", NeuronX)
    setattr(thismodule, "NeuronY", NeuronY)
    # print("classes NeuronX and NeuronY reassigned")


# reassign classes when the module is loaded
neurontype_random_reassignment()


def getting_started():
    """
    simple demo to get started

    Returns:

    """
    from neurodynex3.tools import input_factory

    # create an input current
    input_current = input_factory.get_step_current(50, 150, 1. * b2.ms, 0.5 * b2.pA)

    # get an instance of class NeuronX
    a_neuron_of_type_X = NeuronX()
    # simulate it and get the state variables
    state_monitor = a_neuron_of_type_X.run(input_current, 200 * b2.ms)
    # plot state vs. time
    plot_data(state_monitor, title="Neuron of Type X")

    # get an instance of class NeuronY
    a_neuron_of_type_Y = NeuronY()
    state_monitor = a_neuron_of_type_Y.run(input_current, 200 * b2.ms)
    plot_data(state_monitor, title="Neuron of Type Y")


if __name__ == "__main__":
    getting_started()
