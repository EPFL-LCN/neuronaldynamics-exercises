"""
Implementation of the Brunel 2000 network:
sparsely connected network of identical LIF neurons (Model A).
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
from brian2 import NeuronGroup, Synapses, PoissonInput
from brian2.monitors import StateMonitor, SpikeMonitor, PopulationRateMonitor
from random import sample
from neurodynex3.tools import plot_tools
from numpy import random
import matplotlib.pyplot as plt


# Default parameters of a single LIF neuron
V_REST = 0. * b2.mV
V_RESET = +10. * b2.mV
FIRING_THRESHOLD = +20. * b2.mV
MEMBRANE_TIME_SCALE = 20. * b2.ms
ABSOLUTE_REFRACTORY_PERIOD = 2.0 * b2.ms
# Default parameters of the network
SYNAPTIC_WEIGHT_W0 = 0.1 * b2.mV
# note: w_ee = w_ei = w0 and w_ie=w_ii = -g*w0
RELATIVE_INHIBITORY_STRENGTH_G = 4.  # balanced
CONNECTION_PROBABILITY_EPSILON = 0.1
SYNAPTIC_DELAY = 1.5 * b2.ms
POISSON_INPUT_RATE = 13. * b2.Hz
N_POISSON_INPUT = 1000

b2.defaultclock.dt = 0.05 * b2.ms


def simulate_brunel_network(
        N_Excit=5000,
        N_Inhib=None,
        N_extern=N_POISSON_INPUT,
        connection_probability=CONNECTION_PROBABILITY_EPSILON,
        w0=SYNAPTIC_WEIGHT_W0,
        g=RELATIVE_INHIBITORY_STRENGTH_G,
        synaptic_delay=SYNAPTIC_DELAY,
        poisson_input_rate=POISSON_INPUT_RATE,
        w_external=None,
        v_rest=V_REST,
        v_reset=V_RESET,
        firing_threshold=FIRING_THRESHOLD,
        membrane_time_scale=MEMBRANE_TIME_SCALE,
        abs_refractory_period=ABSOLUTE_REFRACTORY_PERIOD,
        monitored_subset_size=100,
        random_vm_init=False,
        sim_time=100.*b2.ms):
    """
    Fully parametrized implementation of a sparsely connected network of LIF neurons (Brunel 2000)

    Args:
        N_Excit (int): Size of the excitatory popluation
        N_Inhib (int): optional. Size of the inhibitory population.
            If not set (=None), N_Inhib is set to N_excit/4.
        N_extern (int): optional. Number of presynaptic excitatory poisson neurons. Note: if set to a value,
            this number does NOT depend on N_Excit and NOT depend on connection_probability (this is different
            from the book and paper. Only if N_extern is set to 'None', then N_extern is computed as
            N_Excit*connection_probability.
        connection_probability (float): probability to connect to any of the (N_Excit+N_Inhib) neurons
            CE = connection_probability*N_Excit
            CI = connection_probability*N_Inhib
            Cexternal = N_extern
        w0 (float): Synaptic strength J
        g (float): relative importance of inhibition. J_exc = w0. J_inhib = -g*w0
        synaptic_delay (Quantity): Delay between presynaptic spike and postsynaptic increase of v_m
        poisson_input_rate (Quantity): Poisson rate of the external population
        w_external (float): optional. Synaptic weight of the excitatory external poisson neurons onto all
            neurons in the network. Default is None, in that case w_external is set to w0, which is the
            standard value in the book and in the paper Brunel2000.
            The purpose of this parameter is to see the effect of external input in the
            absence of network feedback(setting w0 to 0mV and w_external>0).
        v_rest (Quantity): Resting potential
        v_reset (Quantity): Reset potential
        firing_threshold (Quantity): Spike threshold
        membrane_time_scale (Quantity): tau_m
        abs_refractory_period (Quantity): absolute refractory period, tau_ref
        monitored_subset_size (int): nr of neurons for which a VoltageMonitor is recording Vm
        random_vm_init (bool): if true, the membrane voltage of each neuron is initialized with a
            random value drawn from Uniform(v_rest, firing_threshold)
        sim_time (Quantity): Simulation time

    Returns:
        (rate_monitor, spike_monitor, voltage_monitor, idx_monitored_neurons)
        PopulationRateMonitor: Rate Monitor
        SpikeMonitor: SpikeMonitor for ALL (N_Excit+N_Inhib) neurons
        StateMonitor: membrane voltage for a selected subset of neurons
        list: index of monitored neurons. length = monitored_subset_size
    """
    if N_Inhib is None:
        N_Inhib = int(N_Excit/4)
    if N_extern is None:
        N_extern = int(N_Excit*connection_probability)
    if w_external is None:
        w_external = w0

    J_excit = w0
    J_inhib = -g*w0

    lif_dynamics = """
    dv/dt = -(v-v_rest) / membrane_time_scale : volt (unless refractory)"""

    network = NeuronGroup(
        N_Excit+N_Inhib, model=lif_dynamics,
        threshold="v>firing_threshold", reset="v=v_reset", refractory=abs_refractory_period,
        method="linear")
    if random_vm_init:
        network.v = random.uniform(v_rest/b2.mV, high=firing_threshold/b2.mV, size=(N_Excit+N_Inhib))*b2.mV
    else:
        network.v = v_rest
    excitatory_population = network[:N_Excit]
    inhibitory_population = network[N_Excit:]

    exc_synapses = Synapses(excitatory_population, target=network, on_pre="v += J_excit", delay=synaptic_delay)
    exc_synapses.connect(p=connection_probability)

    inhib_synapses = Synapses(inhibitory_population, target=network, on_pre="v += J_inhib", delay=synaptic_delay)
    inhib_synapses.connect(p=connection_probability)

    external_poisson_input = PoissonInput(target=network, target_var="v", N=N_extern,
                                          rate=poisson_input_rate, weight=w_external)

    # collect data of a subset of neurons:
    monitored_subset_size = min(monitored_subset_size, (N_Excit+N_Inhib))
    idx_monitored_neurons = sample(range(N_Excit+N_Inhib), monitored_subset_size)
    rate_monitor = PopulationRateMonitor(network)
    # record= some_list is not supported? :-(
    spike_monitor = SpikeMonitor(network, record=idx_monitored_neurons)
    voltage_monitor = StateMonitor(network, "v", record=idx_monitored_neurons)

    b2.run(sim_time)
    return rate_monitor, spike_monitor, voltage_monitor, idx_monitored_neurons


def getting_started():
    """
        A simple example to get started
    """
    rate_monitor, spike_monitor, voltage_monitor, monitored_spike_idx = simulate_brunel_network(
        N_Excit=2000, sim_time=800. * b2.ms)
    plot_tools.plot_network_activity(rate_monitor, spike_monitor, voltage_monitor,
                                     spike_train_idx_list=monitored_spike_idx, t_min=0.*b2.ms,
                                     N_highlighted_spiketrains=3, avg_window_width=1. * b2.ms)
    plt.show()


def _demo_emergence_of_oscillation():
    poisson_rate = 18 * b2.Hz
    g = 2.5

    rate_monitor, spike_monitor, voltage_monitor, monitored_spike_idx = \
        simulate_brunel_network(N_Excit=6000, random_vm_init=True, poisson_input_rate=poisson_rate,
                                g=g, sim_time=300. * b2.ms, monitored_subset_size=50)
    plot_tools.plot_network_activity(rate_monitor, spike_monitor, voltage_monitor,
                                     spike_train_idx_list=monitored_spike_idx, t_min=0*b2.ms)
    plot_tools.plot_network_activity(rate_monitor, spike_monitor, voltage_monitor,
                                     spike_train_idx_list=monitored_spike_idx, t_max=50*b2.ms)
    plot_tools.plot_network_activity(rate_monitor, spike_monitor, voltage_monitor,
                                     spike_train_idx_list=monitored_spike_idx, t_min=250*b2.ms)
    plt.show()


def _some_example_calls_and_tests():
    from neurodynex3.tools import spike_tools
    poisson_rate = 35*b2.Hz
    g = 4
    CE = 5000

    delta_t = 0.1 * b2.ms
    delta_f = 5. * b2.Hz
    T_init = 100 * b2.ms
    k = 9

    f_max = 1./(2. * delta_t)
    N_samples = 2. * f_max / delta_f
    T_signal = N_samples * delta_t
    T_sim = k * T_signal + T_init

    print("Start simulation. T_sim={}, T_signal={}. N_samples={}".format(T_sim, T_signal, N_samples))
    b2.defaultclock.dt = delta_t
    stime = T_sim + (10 + k) * b2.defaultclock.dt  # add a few extra samples (solves rounding issues)
    rate_monitor, spike_monitor, voltage_monitor, monitored_spike_idx = \
        simulate_brunel_network(
            N_Excit=CE, poisson_input_rate=poisson_rate, g=g, sim_time=stime)

    plot_tools.plot_network_activity(rate_monitor, spike_monitor, voltage_monitor,
                                     spike_train_idx_list=monitored_spike_idx, t_min=0*b2.ms)
    plot_tools.plot_network_activity(rate_monitor, spike_monitor, voltage_monitor,
                                     spike_train_idx_list=monitored_spike_idx, t_min=T_sim - 80*b2.ms)
    spike_stats = spike_tools.get_spike_train_stats(spike_monitor, window_t_min=150.*b2.ms)
    plot_tools.plot_ISI_distribution(spike_stats, hist_nr_bins=77, xlim_max_ISI=100*b2.ms)

    #     # Power Spectrum
    pop_freqs, pop_ps, average_population_rate = \
        spike_tools.get_population_activity_power_spectrum(
            rate_monitor, delta_f, k, T_init, subtract_mean_activity=True)

    plot_tools.plot_population_activity_power_spectrum(pop_freqs, pop_ps, 1000*b2.Hz, average_population_rate)
    plt.show()

    freq, mean_ps, all_ps, mean_firing_rate, all_mean_firing_freqs = \
        spike_tools.get_averaged_single_neuron_power_spectrum(
            spike_monitor, sampling_frequency=1./delta_t, window_t_min=100.*b2.ms,
            window_t_max=T_sim,  subtract_mean=False, nr_neurons_average=200)
    print("plot_spike_train_power_spectrum")
    plot_tools.plot_spike_train_power_spectrum(freq, mean_ps, all_ps, 1000 * b2.Hz,
                                               mean_firing_freqs_per_neuron=all_mean_firing_freqs,
                                               nr_highlighted_neurons=2)
    plt.show()
    print("done")


if __name__ == "__main__":
    getting_started()
