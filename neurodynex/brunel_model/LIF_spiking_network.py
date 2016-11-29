"""
Implementation of the Brunel 2000 network:
sparsely connected network of identical LIF neurons (Model A).
"""
import brian2 as b2
from brian2 import NeuronGroup, Synapses, PoissonInput
from brian2.monitors import StateMonitor, SpikeMonitor, PopulationRateMonitor
from random import sample
import numpy
import matplotlib.pyplot as plt


# Default parameters of a single LIF neuron
V_REST = 0. * b2.mV
V_RESET = +10. * b2.mV
FIRING_THRESHOLD = +20. * b2.mV
MEMBRANE_TIME_SCALE = 20. * b2.ms
ABSOLUTE_REFRACTORY_PERIOD = 2.0 * b2.ms
POISSON_INPUT_RATE = 12. * b2.Hz
# Default parameters of the network
SYNAPTIC_WEIGHT_W0 = 0.1 * b2.mV
# note: w_ee = w_ei = w0 and w_ie=w_ii = -g*w0
RELATIVE_INHIBITORY_STRENGTH_G = 4.  # balanced
CONNECTION_PROBABILITY_EPSILON = 0.1
SYNAPTIC_DELAY = 1.5*b2.ms
N_POISSON_INPUT = 1000

b2.defaultclock.dt = 0.07*b2.ms


def simulate_brunel_network(
        N_Excit=2000,
        N_Inhib=None,
        N_extern=N_POISSON_INPUT,
        connection_probability=CONNECTION_PROBABILITY_EPSILON,
        w0=SYNAPTIC_WEIGHT_W0,
        g=RELATIVE_INHIBITORY_STRENGTH_G,
        synaptic_delay=SYNAPTIC_DELAY,
        poisson_input_rate=POISSON_INPUT_RATE,
        v_rest=V_REST,
        v_reset=V_RESET,
        firing_threshold=FIRING_THRESHOLD,
        membrane_time_scale=MEMBRANE_TIME_SCALE,
        abs_refractory_period=ABSOLUTE_REFRACTORY_PERIOD,
        monitored_subset_size=100,
        sim_time=500.*b2.ms):
    """
    Implementation of a sparsely connected network of LIF neurons (Brunel 2000)

    Args:
        N_Excit (int): Size of the excitatory popluation
        N_Inhib (int): optional. Size of the inhibitory population.
            If not set (=None), N_Inhib is set to N_excit/4.
        N_extern (int): optional. Nr of presynaptic excitatory poisson neurons. Note this number does
            NOT depend on N_Excit and NOT depend on connection_probability (unlike in the orig paper)
        connection_probability (float): probability to connect to any of the (N_Excit+N_Inhib) neurons
            CE = connection_probability*N_Excit
            CI = connection_probability*N_Inhib
            Cexternal = N_extern
        w0 (float): Synaptic strength J
        g (float): relative importance of inhibition. J_exc = w0. J_inhib = -g*w0
        synaptic_delay (Quantity): Delay
        poisson_input_rate (Quantity): Poisson rate
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
        N_extern = N_Excit

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

    exc_synapses = Synapses(excitatory_population, target=network,
                            on_pre="v += J_excit", delay=synaptic_delay)

    exc_synapses.connect(p=connection_probability)

    inhib_synapses = Synapses(inhibitory_population, target=network,
                              on_pre="v += J_inhib", delay=synaptic_delay)
    inhib_synapses.connect(p=connection_probability)

    external_poisson_input = PoissonInput(target=network, target_var="v", N=N_extern,
                                          rate=poisson_input_rate, weight=J_excit)

    # collect data of a subset of neurons:
    monitored_subset_size = min(monitored_subset_size, (N_Excit+N_Inhib))
    idx_monitored_neurons = sample(xrange(N_Excit+N_Inhib), monitored_subset_size)
    rate_monitor = PopulationRateMonitor(network)
    # record= some_list is not supported? :-(
    spike_monitor = SpikeMonitor(network, record=idx_monitored_neurons)
    voltage_monitor = StateMonitor(network, "v", record=idx_monitored_neurons)

    b2.run(sim_time)
    return rate_monitor, spike_monitor, voltage_monitor, idx_monitored_neurons


def plot_network_activity(rate_monitor, spike_monitor, voltage_monitor=None, spike_train_idx_list=None,
                          t_min=None, t_max=None, highlight_3_spiketrains=True):
    """
    visualizes the results of the network simulation

    Args:
        rate_monitor (PopulationRateMonitor): rate of the population
        spike_monitor (SpikeMonitor): spike trains of individual neurons
        voltage_monitor (StateMonitor): optional. voltage traces of some (defined in spike_train_idx_list) neurons
        spike_train_idx_list (list): optional. If no list is provided, all spikes in the spike_monitor are plotted
        t_min (Quantity): optional. lower bound of the plotted time interval.
            if t_min is None, it is either set to 0 or to t_max - 150ms
        t_max (Quantity): optional. upper bound of the plotted time interval.
            if t_max is None, it is set to the timestamp of the last spike in
        highlight_3_spiketrains (bool): if true, 3 spike trains are visually highlighted

    """
    assert isinstance(rate_monitor, b2.PopulationRateMonitor), \
        "rate_monitor  is not of type PopulationRateMonitor"
    assert isinstance(spike_monitor, b2.SpikeMonitor), \
        "spike_monitor is not of type SpikeMonitor"
    assert (voltage_monitor is None) or (isinstance(voltage_monitor, b2.StateMonitor)), \
        "voltage_monitor is not of type StateMonitor"
    assert (spike_train_idx_list is None) or (isinstance(spike_train_idx_list, list)), \
        "spike_train_idx_list is not of type list"

    all_spike_trains = spike_monitor.spike_trains()
    if spike_train_idx_list is None:
        if voltage_monitor is not None:
            # if no index list is provided use the one from the voltage monitor
            spike_train_idx_list = numpy.sort(voltage_monitor.record)
        else:
            # no index list AND no voltage monitor: plot all spike trains
            spike_train_idx_list = numpy.sort(all_spike_trains.keys())
        if len(spike_train_idx_list) > 500:
            # avoid slow plotting of a large set
            spike_train_idx_list = spike_train_idx_list[:500]

    # get a reasonable default interval
    if t_max is None:
        t_max = max(spike_monitor.t/b2.ms)
    else:
        t_max = t_max/b2.ms
    if t_min is None:
        t_min = max(0., t_max-120.)  # if none, plot at most the last 120ms
    else:
        t_min = t_min / b2.ms

    fig = None
    ax_raster = None
    ax_rate = None
    ax_voltage = None
    if voltage_monitor is None:
        fig, (ax_raster, ax_rate) = plt.subplots(2, 1, sharex=True)
    else:
        fig, (ax_raster, ax_rate, ax_voltage) = plt.subplots(3, 1, sharex=True)

    nr_neurons = len(spike_train_idx_list)
    highlighted_neurons_i = [int(nr_neurons*v) for v in [.25, .5, .75]]
    highlighted_neurons_c = ["g", [.9, .2, .2], "b"]
    highlighted_neurons_line_width = [.8, 1., .8]

    # nested helpers to plot the parts, note that they use parameters defined outside.
    def get_spike_train_ts_indices(spike_train):
        """
        Helper. Extracts the spikes within the time window from the spike train
        """
        ts = spike_train/b2.ms
        spike_within_time_window = (ts >= t_min) & (ts <= t_max)
        idx_spikes = numpy.where(spike_within_time_window)
        ts_spikes = ts[idx_spikes]
        return idx_spikes, ts_spikes

    def plot_raster():
        """
        Helper. Plots the spike trains of the spikes in spike_train_idx_list
        """
        neuron_counter = 0
        for neuron_index in spike_train_idx_list:
            idx_spikes, ts_spikes = get_spike_train_ts_indices(all_spike_trains[neuron_index])
            ax_raster.scatter(ts_spikes, neuron_counter * numpy.ones(ts_spikes.shape), marker=".", c="k", s=15, lw=0)
            neuron_counter += 1

    def highlight_raster():
        """
        Helper. Highlights three spike trains
        """
        for i in range(len(highlighted_neurons_i)):
            raster_plot_index = highlighted_neurons_i[i]
            population_index = spike_train_idx_list[raster_plot_index]
            idx_spikes, ts_spikes = get_spike_train_ts_indices(all_spike_trains[population_index])
            ax_raster.axhline(y=raster_plot_index, linewidth=1, linestyle="-", color=[.85, .85, .85])
            ax_raster.scatter(
                ts_spikes, raster_plot_index * numpy.ones(ts_spikes.shape),
                marker=".", c=highlighted_neurons_c[i], s=70, lw=0)
        ax_raster.set_ylabel("neuron #")
        ax_raster.set_title("Raster Plot (random subset)", fontsize=12)

    def plot_population_activity():
        """
        Helper. Plots the population rate and a mean
        """
        ts = rate_monitor.t / b2.ms
        idx_rate = (ts >= t_min) & (ts <= t_max)
        # ax_rate.plot(ts[idx_rate],rate_monitor.rate[idx_rate]/b2.Hz, ".k", markersize=2)
        smoothed_rates = rate_monitor.smooth_rate(window="flat", width=1.5*b2.ms)/b2.Hz
        ax_rate.plot(ts[idx_rate], smoothed_rates[idx_rate])
        ax_rate.set_ylabel("A(t) [Hz]")
        ax_rate.set_title("Population Activity", fontsize=12)

    def plot_voltage_traces():
        """
        Helper. Plots three voltage traces
        """
        if voltage_monitor is not None:
            ts = voltage_monitor.t/b2.ms
            idx_voltage = (ts >= t_min) & (ts <= t_max)
            for i in range(len(highlighted_neurons_i)):
                raster_plot_index = highlighted_neurons_i[i]
                population_index = spike_train_idx_list[raster_plot_index]
                ax_voltage.plot(
                    ts[idx_voltage], voltage_monitor[population_index].v[idx_voltage]/b2.mV,
                    c=highlighted_neurons_c[i], lw=highlighted_neurons_line_width[i])
                ax_voltage.set_ylabel("V(t) [mV]")
                ax_voltage.set_title("Random Neuron Voltage Traces", fontsize=12)

    plot_raster()
    highlight_raster()
    plot_population_activity()
    plot_voltage_traces()
    plt.xlabel("t [ms]")
    plt.show()


def getting_started():
    """
        A simple example to get started
    """
    rate_monitor, spike_monitor, voltage_monitor, monitored_spike_idx = simulate_brunel_network(
        N_Excit=250, sim_time=100. * b2.ms, monitored_subset_size=7)
    plot_network_activity(rate_monitor, spike_monitor, voltage_monitor,
                          spike_train_idx_list=monitored_spike_idx, t_min=0.*b2.ms, highlight_3_spiketrains=True)

if __name__ == "__main__":
    # getting_started()
    # Test/debug
    # rate_monitor, voltage_monitor, spike_monitor, monitored_spike_idx = simulate_brunel_network(
        # N_Excit=100, N_extern=1000, g=2.5, poisson_input_rate=12*b2.Hz, sim_time=100.*b2.ms, monitored_subset_size=100)

    # SR
    # rate_monitor, voltage_monitor, spike_monitor, monitored_spike_idx = simulate_brunel_network(
    #     N_Excit=6000, N_extern=1000, g=2.5, poisson_input_rate=9*b2.Hz, sim_time=500.*b2.ms, monitored_subset_size=100)

    # SI fast
    # rate_monitor, voltage_monitor, spike_monitor, monitored_spike_idx = simulate_brunel_network(
    #     N_Excit=6000, N_extern=1000, g=8., poisson_input_rate=50*b2.Hz, sim_time=500.*b2.ms, monitored_subset_size=100)


    # AI
    rate_monitor, spike_monitor, voltage_monitor, monitored_spike_idx = \
        simulate_brunel_network(N_Excit=6000, N_extern=1000, g=6.,
                                poisson_input_rate=25*b2.Hz, sim_time=500.*b2.ms,
                                monitored_subset_size=100)
    plot_network_activity(
        rate_monitor, spike_monitor, voltage_monitor,
        spike_train_idx_list=monitored_spike_idx, highlight_3_spiketrains=True)

    # rate_monitor, spike_monitor, voltage_monitor, monitored_spike_idx = simulate_brunel_network(N_Excit=200, sim_time=200. * b2.ms)
    #
    # plot_network_activity(rate_monitor, spike_monitor, voltage_monitor, t_min=20.*b2.ms, spike_train_idx_list=monitored_spike_idx)
    # # plot_network_activity(rate_monitor, monitored_spike_idx, spike_monitor, voltage_monitor, highlight_3_spiketrains=False)
    #

