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


def get_step_curr(I_tstart=20, I_tend=270, I_amp=.5):
    """Returns a pA step current TimedArray.

    Args:
        I_tstart (float, optional): start of current step [ms]
        I_tend (float, optional): start of end step [ms]
        I_amp (float, optional): amplitude of current step [pA]

    Returns:
        StateMonitor: Brian2 StateMonitor with input current (I) and
        voltage (V) recorded
    """

    # 1ms sampled step current
    tmp = np.zeros(I_tend+1) * b2.uamp
    tmp[int(I_tstart):int(I_tend)] = I_amp * b2.pamp

    # This relies on the property of TimedArrays that
    # the returned value is equal to the final array
    # value of the passed array. Here, the step
    # will be 0 for all times > I_tend.
    return b2.TimedArray(tmp, dt=1.*b2.ms)


def plot_data(rec, title=None, show=False):
    """Plots a TimedArray for values I, v and w

    Args:
        rec (TimedArray): the data to plot
        title (string, optional): plot title to display
        show (bool, optional): call plt.show for the plot

    Returns:
        StateMonitor: Brian2 StateMonitor with input current (I) and
        voltage (V) recorded
    """

    (t, v, w, I) = rec_to_tuple(rec)

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

    if show:
        plt.show()


def get_spiketimes(t, v, v_th=0.5, do_plot=False):
    """Returns numpy.ndarray of spike times, for a given time and
    voltage series.

    Args:
        t (numpy.ndarray): time dimension of timeseries [ms]
        v (numpy.ndarray): voltage dimension of timeseries [mV]
        v_th (float, optional): threshold voltage for spike detection [mV]
        do_plot (bool, optional): plot the results

    Returns:
        np.ndarray: detected spike times
    """

    v_above_th = v > v_th
    idx = np.nonzero((v_above_th[:-1] == 0) & (v_above_th[1:] == 1))

    return t[idx[0]+1]


def rec_to_tuple(rec):
    """Extracts a tuple of numpy arrays from a brian2 StateMonitor.

    Args:
        rec (StateMonitor): state monitor with v, w, I recorded

    Returns:
        tuple: (t, v, w, I) tuple of numpy.ndarrays
    """

    # get data from rec
    t = rec.t / b2.ms
    v = rec.v[0] / b2.mV
    w = rec.w[0] / b2.mV
    I = rec.I[0] / b2.pA

    return (t, v, w, I)


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

    def make_neuron(self):
        """Abstract function, which creates neuron attribute for this class."""

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

    def step(self, t_end=300., I_tstart=20,
             I_tend=270, I_amp=.5, do_plot=True, show=True):
        """Runs the neuron for a step current and plots the data.

        Args:
            t_end (float, optional): the simulation time of the model [ms]
            I_tstart (float, optional): start of current step [ms]
            I_tend (float, optional): start of end step [ms]
            I_amp (float, optional): amplitude of current step [nA]
            do_plot (bool, optional): plot the resulting simulation
            show (bool, optional): call plt.show for the plot

        Returns:
            StateMonitor: Brian2 StateMonitor with input current (I) and
            voltage (V) recorded
        """

        curr = get_step_curr(I_tstart, I_tend, I_amp)
        rec = self.run(curr, t_end)

        if do_plot:
            plot_data(
                rec,
                title="%s" % self.__class__.__name__,
                show=show,
            )

        return rec_to_tuple(rec)

    def get_rate(self, I_amp, t_end=1000., do_plot=False):
        """Return the firing rate under a current step.

        Args:
            NeuronClass (type): Subclass of neurons.AbstractNeuron
            I_amp (float): Amplitude of voltage step
            t_end (float): Length of simulation
            do_plot (bool, optional): plot the results

        Returns:
            float: firing rate of neuron
        """

        (t, v, w, I) = self.step(
            t_end=t_end,
            I_amp=I_amp,
            I_tstart=100,
            I_tend=t_end,
            do_plot=do_plot,
            show=False,
        )

        st = get_spiketimes(t, v)

        if do_plot:
            for s in st:
                plt.subplot(311)
                plt.plot(
                    [s, s],
                    [np.min(v), np.max(v)],
                    c='#ff0000'
                )
            plt.show()

        # if no spikes or 1 spike are detected
        if len(st) < 2:
            return 0.0

        isi = st[1:]-st[:-1]

        # rate in Hz (isi is in ms)
        f = 1000.0 / isi.mean()

        return f


class NeuronTypeOne(NeuronAbstract):

    def make_neuron(self):
        """Sets the self.neuron attribute."""

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


class NeuronTypeTwo(NeuronAbstract):

    def make_neuron(self):
        """Sets the self.neuron attribute."""

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
