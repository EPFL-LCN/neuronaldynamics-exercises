"""
Implementation of the Adaptive Exponential Integrate-and-Fire model.

See Neuronal Dynamics
`Chapter 6 Section 1 <http://neuronaldynamics.epfl.ch/online/Ch6.S1.html>`_

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
import neurodynex.tools.input_factory as input_factory

b2.defaultclock.dt = 0.01 * b2.ms

# default values. (see Table 6.1, Initial Burst)
# http://neuronaldynamics.epfl.ch/online/Ch6.S2.html#Ch6.F3
MEMBRANE_TIME_SCALE_tau_m = 5 * b2.ms
MEMBRANE_RESISTANCE_R = 500 * b2.Mohm
V_REST = -70.0 * b2.mV
V_RESET = -51.0 * b2.mV
RHEOBASE_THRESHOLD_v_rh = -50.0 * b2.mV
SHARPNESS_delta_T = 2.0 * b2.mV
ADAPTATION_VOLTAGE_COUPLING_a = 0.5 * b2.nS
ADAPTATION_TIME_CONSTANT_tau_w = 100.0 * b2.ms
SPIKE_TRIGGERED_ADAPTATION_INCREMENT_b = 7.0 * b2.pA

# a technical threshold to tell the algorithm when to reset vm to v_reset
FIRING_THRESHOLD_v_spike = -30. * b2.mV


# This function implement Adaptive Exponential Leaky Integrate-And-Fire neuron model
def simulate_AdEx_neuron(
        tau_m=MEMBRANE_TIME_SCALE_tau_m,
        R=MEMBRANE_RESISTANCE_R,
        v_rest=V_REST,
        v_reset=V_RESET,
        v_rheobase=RHEOBASE_THRESHOLD_v_rh,
        a=ADAPTATION_VOLTAGE_COUPLING_a,
        b=SPIKE_TRIGGERED_ADAPTATION_INCREMENT_b,
        v_spike=FIRING_THRESHOLD_v_spike,
        delta_T=SHARPNESS_delta_T,
        tau_w=ADAPTATION_TIME_CONSTANT_tau_w,
        I_stim=input_factory.get_zero_current(),
        simulation_time=200 * b2.ms):
    r"""
    Implementation of the AdEx model with a single adaptation variable w.

    The Brian2 model equations are:

    .. math::

        \frac{dv}{dt} = (-(v-v_rest) +delta_T*exp((v-v_rheobase)/delta_T)+ R * I_stim(t,i) - R * w)/(tau_m) : volt \\
        \frac{dw}{dt} = (a*(v-v_rest)-w)/tau_w : amp

    Args:
        tau_m (Quantity): membrane time scale
        R (Quantity): membrane restistance
        v_rest (Quantity): resting potential
        v_reset (Quantity): reset potential
        v_rheobase (Quantity): rheobase threshold
        a (Quantity): Adaptation-Voltage coupling
        b (Quantity): Spike-triggered adaptation current (=increment of w after each spike)
        v_spike (Quantity): voltage threshold for the spike condition
        delta_T (Quantity): Sharpness of the exponential term
        tau_w (Quantity): Adaptation time constant
        I_stim (TimedArray): Input current
        simulation_time (Quantity): Duration for which the model is simulated

    Returns:
        (state_monitor, spike_monitor):
        A b2.StateMonitor for the variables "v" and "w" and a b2.SpikeMonitor
    """

    v_spike_str = "v>{:f}*mvolt".format(v_spike / b2.mvolt)

    # EXP-IF
    eqs = """
        dv/dt = (-(v-v_rest) +delta_T*exp((v-v_rheobase)/delta_T)+ R * I_stim(t,i) - R * w)/(tau_m) : volt
        dw/dt=(a*(v-v_rest)-w)/tau_w : amp
        """

    neuron = b2.NeuronGroup(1, model=eqs, threshold=v_spike_str, reset="v=v_reset;w+=b", method="euler")

    # initial values of v and w is set here:
    neuron.v = v_rest
    neuron.w = 0.0 * b2.pA

    # Monitoring membrane voltage (v) and w
    state_monitor = b2.StateMonitor(neuron, ["v", "w"], record=True)
    spike_monitor = b2.SpikeMonitor(neuron)

    # running simulation
    b2.run(simulation_time)
    return state_monitor, spike_monitor


def plot_adex_state(adex_state_monitor):
    """
    Visualizes the state variables: w-t, v-t and phase-plane w-v

    Args:
        adex_state_monitor (StateMonitor): States of "v" and "w"

    """
    plt.subplot(2, 2, 1)
    plt.plot(adex_state_monitor.t / b2.ms, adex_state_monitor.v[0] / b2.mV, lw=2)
    plt.xlabel("t [ms]")
    plt.ylabel("u [mV]")
    plt.title("Membrane potential")
    plt.subplot(2, 2, 2)
    plt.plot(adex_state_monitor.v[0] / b2.mV, adex_state_monitor.w[0] / b2.pA, lw=2)
    plt.xlabel("u [mV]")
    plt.ylabel("w [pAmp]")
    plt.title("Phase plane representation")
    plt.subplot(2, 2, 3)
    plt.plot(adex_state_monitor.t / b2.ms, adex_state_monitor.w[0] / b2.pA, lw=2)
    plt.xlabel("t [ms]")
    plt.ylabel("w [pAmp]")
    plt.title("Adaptation current")
    plt.show()


def getting_started():
    """
    Simple example to get started
    """

    from neurodynex.tools import plot_tools
    current = input_factory.get_step_current(10, 200, 1. * b2.ms, 65.0 * b2.pA)
    state_monitor, spike_monitor = simulate_AdEx_neuron(I_stim=current, simulation_time=300 * b2.ms)
    plot_tools.plot_voltage_and_current_traces(state_monitor, current)
    plot_adex_state(state_monitor)
    print("nr of spikes: {}".format(spike_monitor.count[0]))

if __name__ == "__main__":
    getting_started()
