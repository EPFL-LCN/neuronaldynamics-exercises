"""
This file implements a leaky intergrate-and-fire (LIF) model.
You can inject a step current or sinusoidal current into
neuron using LIF_Step() or LIF_Sinus() methods respectively.

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
from neurodynex3.tools import input_factory, plot_tools
import random
import matplotlib.pyplot as plt

# Neuron model default values
V_REST = -70 * b2.mV
V_RESET = -65 * b2.mV
FIRING_THRESHOLD = -50 * b2.mV
MEMBRANE_RESISTANCE = 10. * b2.Mohm
MEMBRANE_TIME_SCALE = 8. * b2.ms
ABSOLUTE_REFRACTORY_PERIOD = 2.0 * b2.ms


def print_default_parameters():
    """
    Prints the default values
    Returns:

    """
    print("Resting potential: {}".format(V_REST))
    print("Reset voltage: {}".format(V_RESET))
    print("Firing threshold: {}".format(FIRING_THRESHOLD))
    print("Membrane resistance: {}".format(MEMBRANE_RESISTANCE))
    print("Membrane time-scale: {}".format(MEMBRANE_TIME_SCALE))
    print("Absolute refractory period: {}".format(ABSOLUTE_REFRACTORY_PERIOD))


def simulate_LIF_neuron(input_current,
                        simulation_time=5 * b2.ms,
                        v_rest=V_REST,
                        v_reset=V_RESET,
                        firing_threshold=FIRING_THRESHOLD,
                        membrane_resistance=MEMBRANE_RESISTANCE,
                        membrane_time_scale=MEMBRANE_TIME_SCALE,
                        abs_refractory_period=ABSOLUTE_REFRACTORY_PERIOD):
    """Basic leaky integrate and fire neuron implementation.

    Args:
        input_current (TimedArray): TimedArray of current amplitudes. One column per current_injection_location.
        simulation_time (Quantity): Time for which the dynamics are simulated: 5ms
        v_rest (Quantity): Resting potential: -70mV
        v_reset (Quantity): Reset voltage after spike - 65mV
        firing_threshold (Quantity) Voltage threshold for spiking -50mV
        membrane_resistance (Quantity): 10Mohm
        membrane_time_scale (Quantity): 8ms
        abs_refractory_period (Quantity): 2ms

    Returns:
        StateMonitor: Brian2 StateMonitor for the membrane voltage "v"
        SpikeMonitor: Biran2 SpikeMonitor
    """

    # differential equation of Leaky Integrate-and-Fire model
    eqs = """
    dv/dt =
    ( -(v-v_rest) + membrane_resistance * input_current(t,i) ) / membrane_time_scale : volt (unless refractory)"""

    # LIF neuron using Brian2 library
    neuron = b2.NeuronGroup(
        1, model=eqs, reset="v=v_reset", threshold="v>firing_threshold",
        refractory=abs_refractory_period, method="linear")
    neuron.v = v_rest  # set initial value

    # monitoring membrane potential of neuron and injecting current
    state_monitor = b2.StateMonitor(neuron, ["v"], record=True)
    spike_monitor = b2.SpikeMonitor(neuron)
    # run the simulation
    b2.run(simulation_time)
    return state_monitor, spike_monitor


__OBFUSCATION_FACTORS = [543, 622, 9307, 584, 2029, 211]


def _obfuscate_params(param_set):
    """ A helper to _obfuscate_params a parameter vector.

    Args:
        param_set:

    Returns:
        list: obfuscated list
    """
    obfuscated_factors = [__OBFUSCATION_FACTORS[i] * param_set[i] for i in range(6)]
    return obfuscated_factors


def _deobfuscate_params(obfuscated_params):
    """ A helper to deobfuscate a parameter set.

    Args:
        obfuscated_params (list):

    Returns:
        list: de-obfuscated list
    """
    param_set = [obfuscated_params[i] / __OBFUSCATION_FACTORS[i] for i in range(6)]
    return param_set


def get_random_param_set(random_seed=None):
    """
    creates a set of random parameters. All values are constrained to their typical range
    Returns:
        list: a list of (obfuscated) parameters. Use this vector when calling simulate_random_neuron()
    """
    random.seed(random_seed)
    v_rest = (-75. + random.randint(0, 15)) * b2.mV
    v_reset = v_rest + random.randint(-10, +10) * b2.mV
    firing_threshold = random.randint(-40, +5) * b2.mV
    membrane_resistance = random.randint(2, 15) * b2.Mohm
    membrane_time_scale = random.randint(2, 30) * b2.ms
    abs_refractory_period = random.randint(1, 7) * b2.ms
    true_rand_params = [v_rest, v_reset, firing_threshold,
                        membrane_resistance, membrane_time_scale, abs_refractory_period]
    return _obfuscate_params(true_rand_params)


def print_obfuscated_parameters(obfuscated_params):
    """ Print the de-obfuscated values to the console

    Args:
        obfuscated_params:

    Returns:

    """
    true_vals = _deobfuscate_params(obfuscated_params)
    print("Resting potential: {}".format(true_vals[0]))
    print("Reset voltage: {}".format(true_vals[1]))
    print("Firing threshold: {}".format(true_vals[2]))
    print("Membrane resistance: {}".format(true_vals[3]))
    print("Membrane time-scale: {}".format(true_vals[4]))
    print("Absolute refractory period: {}".format(true_vals[5]))


def simulate_random_neuron(input_current, obfuscated_param_set):
    """
    Simulates a LIF neuron with unknown parameters (obfuscated_param_set)
    Args:
        input_current (TimedArray): The current to probe the neuron
        obfuscated_param_set (list): obfuscated parameters

    Returns:
        StateMonitor: Brian2 StateMonitor for the membrane voltage "v"
        SpikeMonitor: Biran2 SpikeMonitor
    """
    vals = _deobfuscate_params(obfuscated_param_set)
    # run the LIF model
    state_monitor, spike_monitor = simulate_LIF_neuron(
        input_current,
        simulation_time=50 * b2.ms,
        v_rest=vals[0],
        v_reset=vals[1],
        firing_threshold=vals[2],
        membrane_resistance=vals[3],
        membrane_time_scale=vals[4],
        abs_refractory_period=vals[5])
    return state_monitor, spike_monitor


def getting_started():
    """
    An example to quickly get started with the LIF module.
    Returns:

    """
    # specify step current
    step_current = input_factory.get_step_current(
        t_start=100, t_end=200, unit_time=b2.ms,
        amplitude=1.2 * b2.namp)
    # run the LIF model
    (state_monitor, spike_monitor) = simulate_LIF_neuron(input_current=step_current, simulation_time=300 * b2.ms)

    # plot the membrane voltage
    plot_tools.plot_voltage_and_current_traces(state_monitor, step_current,
                                               title="Step current", firing_threshold=FIRING_THRESHOLD)
    print("nr of spikes: {}".format(len(spike_monitor.t)))
    plt.show()

    # second example: sinusoidal current. note the higher resolution 0.1 * b2.ms
    sinusoidal_current = input_factory.get_sinusoidal_current(
        500, 1500, unit_time=0.1 * b2.ms,
        amplitude=2.5 * b2.namp, frequency=150 * b2.Hz, direct_current=2. * b2.namp)
    # run the LIF model
    (state_monitor, spike_monitor) = simulate_LIF_neuron(
        input_current=sinusoidal_current, simulation_time=200 * b2.ms)
    # plot the membrane voltage
    plot_tools.plot_voltage_and_current_traces(
        state_monitor, sinusoidal_current, title="Sinusoidal input current", firing_threshold=FIRING_THRESHOLD)
    print("nr of spikes: {}".format(spike_monitor.count[0]))
    plt.show()


if __name__ == "__main__":
    getting_started()
