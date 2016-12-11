
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
import numpy as np
import math


# def get_spike_time(voltage_monitor, spike_threshold):
#     """
#     Detects the spike times in the voltage. The spike time is the value in voltage_monitor.t for
#     which voltage_monitor.v[idx] is above threshold AND voltage_monitor.v[idx-1] is below threshold
#     (crossing from below).
#     Note: currently only the spike times of the first column in voltage_monitor are detected. Matrix-like
#     monitors are not supported.
#     Args:
#         voltage_monitor (StateMonitor): A state monitor with at least the fields "v: and "t"
#         spike_threshold (Quantity): The spike threshold voltage. e.g. -50*b2.mV
#
#     Returns:
#         A list of spike times (Quantity)
#     """
#     assert isinstance(voltage_monitor, b2.StateMonitor), "voltage_monitor is not of type StateMonitor"
#     assert b2.units.fundamentalunits.have_same_dimensions(spike_threshold, b2.volt),\
#         "spike_threshold must be a voltage. e.g. brian2.mV"
#
#     v_above_th = np.asarray(voltage_monitor.v[0] > spike_threshold, dtype=int)
#     diffs = np.diff(v_above_th)
#     boolean_mask = diffs > 0  # cross from below.
#     spike_times = (voltage_monitor.t[1:])[boolean_mask]
#     return spike_times
#
#
# def get_spike_stats(voltage_monitor, spike_threshold):
#     """
#     Detects spike times and computes ISI, mean ISI and firing frequency.
#     Note: meanISI and firing frequency are set to numpy.nan if less than two spikes are detected
#     Note: currently only the spike times of the first column in voltage_monitor are detected. Matrix-like
#     monitors are not supported.
#     Args:
#         voltage_monitor (StateMonitor): A state monitor with at least the fields "v: and "t"
#         spike_threshold (Quantity): The spike threshold voltage. e.g. -50*b2.mV
#
#     Returns:
#         tuple: (nr_of_spikes, spike_times, isi, mean_isi, spike_rate)
#     """
#     spike_times = get_spike_time(voltage_monitor, spike_threshold)
#     isi = np.diff(spike_times)
#     nr_of_spikes = len(spike_times)
#     # init with nan, compute values only if 2 or more spikes are detected
#     mean_isi = np.nan * b2.ms
#     var_isi = np.nan * (b2.ms ** 2)
#     spike_rate = np.nan * b2.Hz
#     if nr_of_spikes >= 2:
#         mean_isi = np.mean(isi)
#         var_isi = np.var(isi)
#         spike_rate = 1. / mean_isi
#     return nr_of_spikes, spike_times, isi, mean_isi, spike_rate, var_isi
#
#
# def pretty_print_spike_train_stats(voltage_monitor, spike_threshold):
#     """
#     Computes and returns the same values as get_spike_stats. Additionally prints these values to the console.
#     Args:
#         voltage_monitor:
#         spike_threshold:
#
#     Returns:
#         tuple: (nr_of_spikes, spike_times, isi, mean_isi, spike_rate)
#     """
#     nr_of_spikes, spike_times, ISI, mean_isi, spike_freq, var_isi = \
#         get_spike_stats(voltage_monitor, spike_threshold)
#     print("nr of spikes: {}".format(nr_of_spikes))
#     print("mean ISI: {}".format(mean_isi))
#     print("ISI variance: {}".format(var_isi))
#     print("spike freq: {}".format(spike_freq))
#     if nr_of_spikes > 20:
#         print("spike times: too many values")
#         print("ISI: too many values")
#     else:
#         print("spike times: {}".format(spike_times))
#         print("ISI: {}".format(ISI))
#     return spike_times, ISI, mean_isi, spike_freq, var_isi


class PopulationSpikeStats:
    """
    Wraps a few spike-train related properties.
    """
    def __init__(self, nr_neurons, nr_spikes, all_ISI, filtered_spike_trains):
        """

        Args:
            nr_neurons:
            nr_spikes:
            mean_isi:
            std_isi:
            all_ISI: list of ISI values (can be used to plot a histrogram)
            filtered_spike_trains the spike trains used to compute the stats. It's a time-window filtered copy of
                the original spike_monitor.all_spike_trains.

        Returns:
            An instance of PopulationSpikeStats
        """
        self._nr_neurons = nr_neurons
        self._nr_spikes = nr_spikes
        self._all_ISI = all_ISI
        self._filtered_spike_trains = filtered_spike_trains

    @property
    def nr_neurons(self):
        """
        Number of neurons in the original population
        """
        return self._nr_neurons

    @property
    def nr_spikes(self):
        """
        Nr of spikes
        """
        return self._nr_spikes

    @property
    def filtered_spike_trains(self):
        """
        a time-window filtered copy of the original spike_monitor.all_spike_trains
        """
        return self._filtered_spike_trains

    @property
    def mean_isi(self):
        """
        Mean Inter Spike Interval
        """
        mean_isi = np.mean(self._all_ISI)*b2.second
        return mean_isi

    @property
    def std_isi(self):
        """
        Standard deviation of the ISI
        """
        std_isi = np.std(self._all_ISI)*b2.second
        return std_isi

    @property
    def all_ISI(self):
        """
        all ISIs in no specific order
        """
        return self._all_ISI

    @property
    def CV(self):
        """
        Coefficient of Variation
        """
        cv = np.nan  # init with nan
        if self.mean_isi > 0.:
            cv = self.std_isi / self.mean_isi
        return cv


def filter_spike_trains(spike_trains, window_t_min=0.*b2.ms, window_t_max=None):
    """
    creates a new dictionary neuron_idx=>spike_times where all spike_times are in the
        half open interval [window_t_min,window_t_max)

    Args:
        spike_trains (dict): a dictionary of spike trains. Typically obtained by
            calling spike_monitor.spike_trains()
        window_t_min (Quantity): Lower bound of the time window: t>=window_t_min. Default is 0ms.
        window_t_max (Quantity): Upper bound of the time window: t<window_t_max.
            Default is None, in which case no upper bound is set.

    Returns:
        a filtered copy of spike_trains
    """
    assert isinstance(spike_trains, dict), \
        "spike_trains is not of type dict"
    nr_neurons = len(spike_trains)
    filtered_spike_trains = dict()
    if (window_t_min == 0.*b2.ms) & (window_t_max is None):
        # print("just copy")
        filtered_spike_trains = spike_trains.copy()
    elif (window_t_max is None):
        # print("only lower bound")
        for i in range(nr_neurons):
            idx = (spike_trains[i] >= window_t_min)
            filtered_spike_trains[i] = spike_trains[i][idx]
    else:
        # print("lower and upper bound")
        for i in range(nr_neurons):
            idx = (spike_trains[i] >= window_t_min) & (spike_trains[i] < window_t_max)
            filtered_spike_trains[i] = spike_trains[i][idx]

    return filtered_spike_trains


def get_spike_train_stats(spike_monitor, window_t_min=0.*b2.ms, window_t_max=None):
    """
    Analyses the spike monitor and returns a PopulationSpikeStats instance.

    Args:
        spike_monitor (SpikeMonitor): Brian2 spike monitor
        window_t_min (Quantity): Lower bound of the time window: t>=window_t_min. The stats are computed
            for spikes within the time window. Default is 0ms
        window_t_max (Quantity): Upper bound of the time window: t<window_t_max. The stats are computed
            for spikes within the time window. Default is None, in which case no upper bound is set.

    Returns:
        PopulationSpikeStats
    """
    assert isinstance(spike_monitor, b2.SpikeMonitor), \
        "spike_monitor is not of type SpikeMonitor"
    filtered_spike_trains = filter_spike_trains(spike_monitor.spike_trains(), window_t_min, window_t_max)
    nr_neurons = len(filtered_spike_trains)
    all_ISI = []
    for i in range(nr_neurons):
        spike_times = filtered_spike_trains[i]/b2.ms
        nr_spikes = len(spike_times)
        if nr_spikes >= 2:
            isi = np.diff(spike_times)
            # maxISI = max(isi)
            # if maxISI > 400:
            #     print(maxISI)
            all_ISI = np.hstack([all_ISI, isi])
    all_ISI = all_ISI*b2.ms
    stats = PopulationSpikeStats(nr_neurons, spike_monitor.num_spikes, all_ISI, filtered_spike_trains)
    return stats


def _spike_train_2_binary_vector(spike_train, vector_length, discretization_dt):
    """
    Convert the time-stamps of the spike_train into a binary vector of the given length.
        Note: if more than one spike fall into the same time bin, only one is counted, surplus spikes are ignored.

    Args:
        spike_train:
        vector_length:
        discretization_dt:

    Returns:
        Discretized spike train: a fixed-length, binary vector.
    """
    vec = np.zeros(vector_length, int)
    idx = spike_train / discretization_dt
    idx = (np.round(idx)).astype(int)
    vec[idx] = 1
    return vec


def _get_spike_train_power_spectrum(spike_train, time_step):
    st = spike_train/b2.ms
    # data = st-(np.mean(st))
    data = st
    ps = np.abs(np.fft.fft(data))**2
    freqs = np.fft.fftfreq(data.size, time_step)
    idx = np.argsort(freqs)
    ps = ps[idx]
    freqs = freqs[idx]
    return ps, freqs


def get_average_power_spectrum(spike_monitor, sampling_frequency, window_t_min=0.*b2.ms, window_t_max=None):
    """
    averaged power-spectrum of spike trains in the time window [window_t_min, window_t_max).
        The power spectrum of every single neuron's spike train is computed. Then the average
        across all single-neuron powers is computed.

    Args:
        spike_monitor (SpikeMonitor) : Brian2 SpikeMonitor
        sampling_frequency (Quantity): sampling frequency used to discretize the spike trains.
        window_t_min (Quantity): Lower bound of the time window: t>=window_t_min. Default is 0ms. Spikes
            before window_t_min are not taken into account (set a lower bound if you want to exclude an initial
            transient in the population activity)
        window_t_max (Quantity): Upper bound of the time window: t<window_t_max. Default is None, in which
            case no upper bound is set.

    Returns:
        freq, mean_ps, all_ps, nyquist_frequency.
    """

    assert isinstance(spike_monitor, b2.SpikeMonitor), \
        "spike_monitor is not of type SpikeMonitor"
    sptrs = filter_spike_trains(spike_monitor.spike_trains(), window_t_min, window_t_max)
    nr_neurons = len(sptrs)
    discretization_dt = 1./sampling_frequency
    if window_t_max is None:
        window_t_max = max(spike_monitor.t)
    vector_length = 1+int(math.ceil((window_t_max-window_t_min)/discretization_dt))
    freq = 0
    all_ps = np.zeros([nr_neurons, vector_length], float)
    for i in range(nr_neurons):
        vec = _spike_train_2_binary_vector(
            sptrs[i]-window_t_min, vector_length, discretization_dt=discretization_dt)
        ps, freq = _get_spike_train_power_spectrum(vec, discretization_dt)
        all_ps[i, :] = ps
    mean_ps = np.mean(all_ps, 0)
    nyquist_frequency = sampling_frequency/2.
    return freq, mean_ps, all_ps, nyquist_frequency


def downsample_population_activity(rate_monitor, downsampling_factor, window_t_min=0.*b2.ms, window_t_max=None):
    # get the population rates within the time window:
    idx = rate_monitor.t >= window_t_min
    if window_t_max is not None:
        idx = idx & (rate_monitor.t < window_t_max)
    rates = rate_monitor.rate[idx]
    # get a multiple of the downsampling factor
    nr_rates = len(rates)
    nr_dropped_samples = nr_rates % downsampling_factor
    if nr_dropped_samples > 0:
        rates = rates[:-nr_dropped_samples]
    nr_rows = len(rates)/downsampling_factor
    r = np.reshape(rates, (nr_rows, downsampling_factor))
    downsampled_rates = np.mean(r, 1)
    return downsampled_rates, nr_dropped_samples


def get_population_activity_power_spectrum(
        rate_monitor, sampling_frequency_upper_bound,
        window_t_min=0.*b2.ms, window_t_max=None, subtract_mean_activity=True):
    downsampling_factor = int(math.ceil((1./rate_monitor.clock.dt)/sampling_frequency_upper_bound))
    if downsampling_factor < 1:
        # int(0.987654321) = 0 would correspond to oversampling
        exc_msg = "sampling frequency is {}, sampling_frequency_upper_bound is {}. Oversampling is not supported."\
            .format(1./rate_monitor.clock.dt, sampling_frequency_upper_bound)
        raise Exception(exc_msg)
    downsampled_fequency = 1. / rate_monitor.clock.dt / downsampling_factor
    nyquist_frequency = downsampled_fequency / 2.

    print("downsampling_factor={}".format(downsampling_factor))
    downsampled_rates, nr_dropped_samples = downsample_population_activity(
        rate_monitor, downsampling_factor, window_t_min, window_t_max)
    if subtract_mean_activity:
        downsampled_rates = downsampled_rates - np.mean(downsampled_rates)
    data = downsampled_rates / b2.Hz
    ps = np.abs(np.fft.fft(data))**2
    freqs = np.fft.fftfreq(data.size, 1./downsampled_fequency)
    idx = np.argsort(freqs)
    ps = ps[idx]
    freqs = freqs[idx]
    return freqs, ps, downsampling_factor, nyquist_frequency
