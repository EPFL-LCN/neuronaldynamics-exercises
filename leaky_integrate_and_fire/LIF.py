"""
***********************************************************
***** Ecole polytechnique federale de Lausanne (EPFL) *****
***** Laboratory of omputational Neuroscience (LCN)  *****
*****               Neuronal Dynamics                 *****
***********************************************************

This file implements a leaky intergrate-and-fire (LIF) model.
You can inject a step current or sinusoidal current into
neuron using LIF_Step() or LIF_Sinus() methods respectively.

In order to know parameters and default values for each method
use symbol ? after the name of method. For example: LIF_Step?

***********************************************************

This file is part of the exercise code repository accompanying
the book: Neuronal Dynamics (see <http://neuronaldynamics.epfl.ch>)
located at <http://github.com/EPFL-LCN/neuronaldynamics-exercises>.

This free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License 2.0 as published by the
Free Software Foundation. You should have received a copy of the
GNU General Public License along with the repository. If not,
see <http://www.gnu.org/licenses/>.

Should you reuse and publish the code for your own purposes,
please cite the book or point to the webpage <http://neuronaldynamics.epfl.ch>.

Wulfram Gerstner, Werner M. Kistler, Richard Naud, and Liam Paninski.
Neuronal Dynamics: From Single Neurons to Networks and Models of Cognition.
Cambridge University Press, 2014."""

from brian2 import *
from brian2.units.stdunits import nA
import pylab as plt

# constants
v_reset = 0.*mV
v_threshold = 1.*mV
R = 1*Mohm
v_rest = 0*mV
tau = 1*ms


def do_plot(rec, v_threshold=1.*mV, **kwargs):

    """Plots the brian recorder 'rec' for values I and v"""

    plt.subplot(211)
    plt.plot(rec.t/ms, rec.v[0]/mV, lw=2)

    plt.plot(
        (rec.t/ms)[[0, -1]],
        [v_threshold/mV, v_threshold/mV],
        'r--', lw=2
    )
    
    plt.xlabel('t [ms]')
    plt.ylabel('v [mV]')
    plt.ylim(0, v_threshold/mV * 1.2)
    plt.grid()

    plt.subplot(212)
    plt.plot(rec.t/ms, rec.I[0]/nA, lw=2)
    plt.xlabel('t [ms]')
    plt.ylabel('I [mV]')
    plt.grid()

    if "title" in kwargs:
        plt.suptitle(kwargs["title"])

    plt.show()


def LIF_Neuron(curr, simtime):

    v_reset_ = "v=%f*volt" % v_reset
    v_threshold_ = "v>%f*volt" % v_threshold

    # differential equation of Leaky Integrate-and-Fire model
    eqs = '''
        dv/dt = ( -(v-v_rest) + R * I ) / tau : volt
        I = curr(t) : amp
    '''

    # LIF neuron using Brian2 library
    IF = NeuronGroup(1, model=eqs, reset=v_reset_, threshold=v_threshold_)
    IF.v = v_rest

    # monitoring membrane potential of neuron and injecting current
    rec = StateMonitor(IF, ['v', 'I'], record=True)

    # run the simulation
    run(simtime)

    return rec


def LIF_Step(I_tstart=20, I_tend=70, I_amp=1.005, tend=100):

    """Run the LIF and give a step current input.

    Parameters:
    tend = 100    (ms) - is the end time of the model
    I_tstart = 20 (ms) - start of current step
    I_tend = 70   (ms) - end of current step
    I_amp = 1.005 (nA) - amplitude of current step
    """

    # 1ms sampled step current
    tmp = numpy.zeros(tend) * nA
    tmp[int(I_tstart):int(I_tend)] = I_amp * nA
    curr = TimedArray(tmp, dt=1.*ms)

    do_plot(LIF_Neuron(curr, tend * ms), title="Step current")


def LIF_Sinus(I_freq=0.1, I_offset=0.5, I_amp=0.5, tend=100, dt=.1):

    """
    Run the LIF for a sinusoidal current

    Parameters:
    tend = 100     (ms) - is the end time of the model
    I_freq = 0.1   (kHz) - frequency of current sinusoidal
    I_offset = 0.5 (nA) - offset current
    I_amp = 0.5    (nA) - amplitude of sin
    """

    # dt sampled sinusoidal function
    t = numpy.arange(0, tend, dt)
    tmp = (I_amp*numpy.sin(2.0*numpy.pi*I_freq*t)+I_offset) * nA
    curr = TimedArray(tmp, dt=dt*ms)

    do_plot(LIF_Neuron(curr, tend * ms), title="Sinusoidal current")
