"""
Exponential Integrate-and-Fire model.
See Neuronal Dynamics, `Chapter 5 Section 2 <http://neuronaldynamics.epfl.ch/online/Ch5.S2.html>`_
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
import numpy
from neurodynex3.tools import input_factory

b2.defaultclock.dt = 0.05 * b2.ms

# default values.
MEMBRANE_TIME_SCALE_tau = 12.0 * b2.ms
MEMBRANE_RESISTANCE_R = 20.0 * b2.Mohm
V_REST = -65.0 * b2.mV
V_RESET = -60.0 * b2.mV
RHEOBASE_THRESHOLD_v_rh = -55.0 * b2.mV
SHARPNESS_delta_T = 2.0 * b2.mV

# a technical threshold to tell the algorithm when to reset vm to v_reset
FIRING_THRESHOLD_v_spike = -30. * b2.mV


def simulate_exponential_IF_neuron(
        tau=MEMBRANE_TIME_SCALE_tau,
        R=MEMBRANE_RESISTANCE_R,
        v_rest=V_REST,
        v_reset=V_RESET,
        v_rheobase=RHEOBASE_THRESHOLD_v_rh,
        v_spike=FIRING_THRESHOLD_v_spike,
        delta_T=SHARPNESS_delta_T,
        I_stim=input_factory.get_zero_current(),
        simulation_time=200 * b2.ms):
    """
    Implements the dynamics of the exponential Integrate-and-fire model

    Args:
        tau (Quantity): Membrane time constant
        R (Quantity): Membrane resistance
        v_rest (Quantity): Resting potential
        v_reset (Quantity): Reset value (vm after spike)
        v_rheobase (Quantity): Rheobase threshold
        v_spike (Quantity) : voltage threshold for the spike condition
        delta_T (Quantity): Sharpness of the exponential term
        I_stim (TimedArray): Input current
        simulation_time (Quantity): Duration for which the model is simulated

    Returns:
        (voltage_monitor, spike_monitor):
        A b2.StateMonitor for the variable "v" and a b2.SpikeMonitor
    """

    eqs = """
    dv/dt = (-(v-v_rest) +delta_T*exp((v-v_rheobase)/delta_T)+ R * I_stim(t,i))/(tau) : volt
    """
    neuron = b2.NeuronGroup(1, model=eqs, reset="v=v_reset", threshold="v>v_spike", method="euler")
    neuron.v = v_rest
    # monitoring membrane potential of neuron and injecting current
    voltage_monitor = b2.StateMonitor(neuron, ["v"], record=True)
    spike_monitor = b2.SpikeMonitor(neuron)

    # run the simulation
    net = b2.Network(neuron, voltage_monitor, spike_monitor)
    net.run(simulation_time)

    return voltage_monitor, spike_monitor


def getting_started():
    """
    A simple example
    """
    import neurodynex3.tools.plot_tools as plot_tools
    input_current = input_factory.get_step_current(t_start=20, t_end=120, unit_time=b2.ms, amplitude=0.8 * b2.namp)
    state_monitor, spike_monitor = simulate_exponential_IF_neuron(
        I_stim=input_current, simulation_time=180 * b2.ms)
    plot_tools.plot_voltage_and_current_traces(
        state_monitor, input_current, title="step current", firing_threshold=FIRING_THRESHOLD_v_spike)
    print("nr of spikes: {}".format(spike_monitor.count[0]))
    plt.show()


def _min_curr_expl():
    from neurodynex3.tools import plot_tools, input_factory

    durations = [1, 2, 5, 10, 20, 50, 100, 200]
    min_amp = [8.6, 4.45, 2., 1.15, .70, .48, 0.43, .4]
    i = 1
    t = durations[i]
    I_amp = min_amp[i] * b2.namp

    input_current = input_factory.get_step_current(
        t_start=10, t_end=10 + t - 1, unit_time=b2.ms, amplitude=I_amp)

    state_monitor, spike_monitor = simulate_exponential_IF_neuron(
        I_stim=input_current, simulation_time=(t + 20) * b2.ms)

    plot_tools.plot_voltage_and_current_traces(
        state_monitor, input_current, title="step current",
        firing_threshold=FIRING_THRESHOLD_v_spike, legend_location=2)
    plt.show()
    print("nr of spikes: {}".format(spike_monitor.count[0]))


if __name__ == "__main__":
    getting_started()
