import brian2 as b2
from brian2 import NeuronGroup, Synapses, PoissonInput, network_operation
from brian2.monitors import StateMonitor, SpikeMonitor, PopulationRateMonitor
from random import sample
from neurodynex.tools import plot_tools
import numpy
import matplotlib.pyplot as plt
import math
from scipy.special import erf
from numpy.fft import rfft, irfft
from math import floor, ceil

from wm_model import *


def create_doc_fig_1():
    # b2.defaultclock.dt = 0.1 * b2.ms
    b2.defaultclock.dt = 0.05 * b2.ms
    rate_monitor_excit, spike_monitor_excit, voltage_monitor_excit, idx_monitored_neurons_excit,\
        rate_monitor_inhib, spike_monitor_inhib, voltage_monitor_inhib, idx_monitored_neurons_inhib\
    = simulate_wm(N_excitatory=1024, N_inhibitory=256,
                  weight_scaling_factor=2.,
                  sim_time=800. * b2.ms, t_stimulus_start=200 * b2.ms, stimulus_center_deg=120)
    plot_tools.plot_network_activity(rate_monitor_excit, spike_monitor_excit, voltage_monitor_excit,
                                     t_min=0. * b2.ms, window_width=5. * b2.ms)
    plot_tools.plot_network_activity(rate_monitor_inhib, spike_monitor_inhib, voltage_monitor_inhib,
                                     t_min=0. * b2.ms, window_width=5. * b2.ms)
    plt.show()


def get_orientation(idx_list, N):
    incr = 360. / N
    orientation = [incr*idx for idx in idx_list]
    return orientation

def get_spike_count(spike_monitor, spike_index_list, t_min, t_max):
    nr_neurons = len(spike_index_list)
    spike_count_list = numpy.zeros(nr_neurons)
    spike_trains = spike_monitor.spike_trains()
    for i in range(nr_neurons):
        neuron_idx = spike_index_list[i]
        spike_count_list[i] = sum((spike_trains[neuron_idx]>=t_min) & (spike_trains[neuron_idx]<(t_max)))
    return spike_count_list


def get_population_theta_trace(spike_monitor, idx_monitored_neurons, t_snapshots, t_window_length):
    theta_pop_at_t = numpy.zeros(len(t_snapshots))
    var_theat_pop = numpy.zeros(len(t_snapshots))
    thetas = get_orientation(idx_monitored_neurons, spike_monitor.source.N)
    for i in range(len(t_snapshots)):
        t_min = t_snapshots[i]
        t_max = t_min + t_window_length
        spike_counts = get_spike_count(spike_monitor, idx_monitored_neurons, t_min, t_max)
        theta_pop_at_t[i] = numpy.average(thetas, weights=spike_counts)
        var_theat_pop[i] = numpy.average((thetas - theta_pop_at_t[i]), weights=spike_counts)
    return theta_pop_at_t, var_theat_pop


def exc_population_vector():
    b2.defaultclock.dt = 0.05 * b2.ms

    t_sim= 300 * b2.ms

    t_stimulus_start = 0 * b2.ms
    t_stimulus_duration = 200 *b2.ms
    stimulus_center_deg = 180


    t_window_length = 50 * b2.ms

    t_snapshots = range(int(floor((t_stimulus_start+t_stimulus_duration)/b2.ms)),
                        int(ceil((t_sim-t_window_length)/b2.ms)), 10)*b2.ms

    nr_repetitions = 1
    list_of_thetas = [[] for i in range(nr_repetitions)]

    for i in range(nr_repetitions):

        rate_monitor_excit, spike_monitor_excit, voltage_monitor_excit, idx_monitored_neurons_excit,\
            rate_monitor_inhib, spike_monitor_inhib, voltage_monitor_inhib, idx_monitored_neurons_inhib\
            = simulate_wm(
                t_stimulus_start=t_stimulus_start, t_stimulus_duration=t_stimulus_duration, stimulus_center_deg=stimulus_center_deg,
                sim_time=t_sim)

    #     fig, ax_raster, ax_rate, ax_voltage = plot_tools.plot_network_activity(rate_monitor_excit, spike_monitor_excit, voltage_monitor_excit,
    #                                      t_min=0. * b2.ms)

        theta_list, var_theta_list, = get_population_theta_trace(
            spike_monitor_excit, idx_monitored_neurons_excit, t_snapshots, t_window_length)
        print(theta_list[0])
        print(var_theta_list[0])
        list_of_thetas[i] = theta_list

    plt.figure()
    for i in range(len(list_of_thetas)):
        theta_trace = list_of_thetas[i]
        plt.plot(t_snapshots/b2.ms,theta_trace-stimulus_center_deg, "-")

    plt.show()




if __name__ == "__main__":
    # create_doc_fig_1()
    exc_population_vector()
