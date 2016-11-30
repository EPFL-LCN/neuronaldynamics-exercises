"""
Implementation of the Brunel 2000 network:
sparsely connected network of identical LIF neurons (Model A).
"""
import brian2 as b2
from brian2 import NeuronGroup, Synapses, PoissonInput
from brian2.monitors import StateMonitor, SpikeMonitor, PopulationRateMonitor
from random import sample
from neurodynex.tools import plot_tools
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

b2.defaultclock.dt = 0.07 * b2.ms


def simulate_brunel_network(
        N_Excit=5000,
        N_Inhib=None,
        N_extern=N_POISSON_INPUT,
        connection_probability=CONNECTION_PROBABILITY_EPSILON,
        w0=SYNAPTIC_WEIGHT_W0,
        g=RELATIVE_INHIBITORY_STRENGTH_G,
        synaptic_delay=SYNAPTIC_DELAY,
        poisson_input_rate=POISSON_INPUT_RATE,
        w_external = None,
        v_rest=V_REST,
        v_reset=V_RESET,
        firing_threshold=FIRING_THRESHOLD,
        membrane_time_scale=MEMBRANE_TIME_SCALE,
        abs_refractory_period=ABSOLUTE_REFRACTORY_PERIOD,
        monitored_subset_size=100,
        sim_time=100.*b2.ms):
    """
    Implementation of a sparsely connected network of LIF neurons (Brunel 2000)

    Args:
        N_Excit (int): Size of the excitatory popluation
        N_Inhib (int): optional. Size of the inhibitory population.
            If not set (=None), N_Inhib is set to N_excit/4.
        N_extern (int): optional. Number of presynaptic excitatory poisson neurons. Note this number does
            NOT depend on N_Excit and NOT depend on connection_probability (unlike in the orig paper).
            If None is provided, then N_extern is set to N_Excit*connection_probability.
        connection_probability (float): probability to connect to any of the (N_Excit+N_Inhib) neurons
            CE = connection_probability*N_Excit
            CI = connection_probability*N_Inhib
            Cexternal = N_extern
        w0 (float): Synaptic strength J
        g (float): relative importance of inhibition. J_exc = w0. J_inhib = -g*w0
        synaptic_delay (Quantity): Delay
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
        sim_time (Quantity): Simulation time

    Returns:
        rate_monitor, spike_monitor, voltage_monitor, idx_monitored_neurons
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
    idx_monitored_neurons = sample(xrange(N_Excit+N_Inhib), monitored_subset_size)
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
        N_Excit=2000, sim_time=500. * b2.ms, poisson_input_rate=15*b2.Hz, monitored_subset_size=100)
    plot_tools.plot_network_activity(rate_monitor, spike_monitor, voltage_monitor,
                          spike_train_idx_list=monitored_spike_idx, t_min=0.*b2.ms, N_highlighted_spiketrains=1)
    plt.show()

if __name__ == "__main__":
    getting_started()
