"""
This the spike_tools submodule provides functions to analyse the
Brian2 SpikeMonitors and Brian2 StateMonitors. The code provided here is
not optimized for performance and there is no guarantee for correctness.

Relevant book chapters:
    - http://neuronaldynamics.epfl.ch/online/Ch19.S2.html#SS1.p6
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
import numpy as np
import math


def get_spike_time(voltage_monitor, spike_threshold):
    """
    Detects the spike times in the voltage. Here, the spike time is DEFINED as the value in
    voltage_monitor.t for which voltage_monitor.v[idx] is above threshold AND
    voltage_monitor.v[idx-1] is below threshold (crossing from below).
    Note: currently only the spike times of the first column in voltage_monitor are detected. Matrix-like
    monitors are not supported.
    Args:
        voltage_monitor (StateMonitor): A state monitor with at least the fields "v: and "t"
        spike_threshold (Quantity): The spike threshold voltage. e.g. -50*b2.mV

    Returns:
        A list of spike times (Quantity)
    """
    assert isinstance(voltage_monitor, b2.StateMonitor), "voltage_monitor is not of type StateMonitor"
    assert b2.units.fundamentalunits.have_same_dimensions(spike_threshold, b2.volt),\
        "spike_threshold must be a voltage. e.g. brian2.mV"

    v_above_th = np.asarray(voltage_monitor.v[0] > spike_threshold, dtype=int)
    diffs = np.diff(v_above_th)
    boolean_mask = diffs > 0  # cross from below.
    spike_times = (voltage_monitor.t[1:])[boolean_mask]
    return spike_times


def get_spike_stats(voltage_monitor, spike_threshold):
    """
    Detects spike times and computes ISI, mean ISI and firing frequency.
    Here, the spike time is DEFINED as the value in
    voltage_monitor.t for which voltage_monitor.v[idx] is above threshold AND
    voltage_monitor.v[idx-1] is below threshold (crossing from below).
    Note: meanISI and firing frequency are set to numpy.nan if less than two spikes are detected
    Note: currently only the spike times of the first column in voltage_monitor are detected. Matrix-like
    monitors are not supported.
    Args:
        voltage_monitor (StateMonitor): A state monitor with at least the fields "v: and "t"
        spike_threshold (Quantity): The spike threshold voltage. e.g. -50*b2.mV

    Returns:
        tuple: (nr_of_spikes, spike_times, isi, mean_isi, spike_rate)
    """
    spike_times = get_spike_time(voltage_monitor, spike_threshold)
    isi = np.diff(spike_times)
    nr_of_spikes = len(spike_times)
    # init with nan, compute values only if 2 or more spikes are detected
    mean_isi = np.nan * b2.ms
    var_isi = np.nan * (b2.ms ** 2)
    spike_rate = np.nan * b2.Hz
    if nr_of_spikes >= 2:
        mean_isi = np.mean(isi)
        var_isi = np.var(isi)
        spike_rate = 1. / mean_isi
    return nr_of_spikes, spike_times, isi, mean_isi, spike_rate, var_isi


def pretty_print_spike_train_stats(voltage_monitor, spike_threshold):
    """
    Computes and returns the same values as get_spike_stats. Additionally prints these values to the console.
    Args:
        voltage_monitor:
        spike_threshold:

    Returns:
        tuple: (nr_of_spikes, spike_times, isi, mean_isi, spike_rate)
    """
    nr_of_spikes, spike_times, ISI, mean_isi, spike_freq, var_isi = \
        get_spike_stats(voltage_monitor, spike_threshold)
    print("nr of spikes: {}".format(nr_of_spikes))
    print("mean ISI: {}".format(mean_isi))
    print("ISI variance: {}".format(var_isi))
    print("spike freq: {}".format(spike_freq))
    if nr_of_spikes > 20:
        print("spike times: too many values")
        print("ISI: too many values")
    else:
        print("spike times: {}".format(spike_times))
        print("ISI: {}".format(ISI))
    return spike_times, ISI, mean_isi, spike_freq, var_isi


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


def filter_spike_trains(spike_trains, window_t_min=0.*b2.ms, window_t_max=None, idx_subset=None):
    """
    creates a new dictionary neuron_idx=>spike_times where all spike_times are in the
        half open interval [window_t_min,window_t_max)

    Args:
        spike_trains (dict): a dictionary of spike trains. Typically obtained by
            calling spike_monitor.spike_trains()
        window_t_min (Quantity): Lower bound of the time window: t>=window_t_min. Default is 0ms.
        window_t_max (Quantity): Upper bound of the time window: t<window_t_max.
            Default is None, in which case no upper bound is set.
        idx_subset (list, optional): a list of neuron indexes (dict keys) specifying a subset of neurons.
            Neurons NOT in the key list are NOT added to the resulting dictionary. Default is None, in which case
            all neurons are added to the resulting list.

    Returns:
        a filtered copy of spike_trains
    """
    assert isinstance(spike_trains, dict), \
        "spike_trains is not of type dict"

    if idx_subset is None:
        idx_subset = spike_trains.keys()

    spike_trains_subset = dict()
    for k in idx_subset:
        spike_trains_subset[k] = spike_trains[k].copy()

    nr_neurons = len(idx_subset)
    filtered_spike_trains = dict()
    if (window_t_min == 0.*b2.ms) & (window_t_max is None):
        # print("just copy")
        filtered_spike_trains = spike_trains_subset
    elif (window_t_max is None):
        # print("only lower bound")
        for i in idx_subset:
            idx = (spike_trains_subset[i] >= window_t_min)
            filtered_spike_trains[i] = spike_trains_subset[i][idx]
    else:
        # print("lower and upper bound")
        for i in idx_subset:
            idx = (spike_trains_subset[i] >= window_t_min) & (spike_trains_subset[i] < window_t_max)
            filtered_spike_trains[i] = spike_trains_subset[i][idx]

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
    idx = np.floor(idx).astype(int)
    vec[idx] = 1
    return vec


def _get_spike_train_power_spectrum(spike_train, delta_t, subtract_mean=False):
    st = spike_train/b2.ms
    if subtract_mean:
        data = st-np.mean(st)
    else:
        data = st
    N_signal = data.size
    ps = np.abs(np.fft.fft(data))**2
    # normalize
    ps = ps * delta_t / N_signal  # TODO: verify: subtract 1 (N_signal-1)?
    freqs = np.fft.fftfreq(N_signal, delta_t)
    ps = ps[:(N_signal/2)]
    freqs = freqs[:(N_signal/2)]
    return ps, freqs


def get_averaged_single_neuron_power_spectrum(spike_monitor, sampling_frequency,
                                              window_t_min, window_t_max,
                                              nr_neurons_average=100, subtract_mean=False):
    """
    averaged power-spectrum of spike trains in the time window [window_t_min, window_t_max).
        The power spectrum of every single neuron's spike train is computed. Then the average
        across all single-neuron powers is computed. In order to limit the compuation time, the
        number of neurons taken to compute the average is limited to nr_neurons_average which defaults to 100

    Args:
        spike_monitor (SpikeMonitor) : Brian2 SpikeMonitor
        sampling_frequency (Quantity): sampling frequency used to discretize the spike trains.
        window_t_min (Quantity): Lower bound of the time window: t>=window_t_min. Spikes
            before window_t_min are not taken into account (set a lower bound if you want to exclude an initial
            transient in the population activity)
        window_t_max (Quantity): Upper bound of the time window: t<window_t_max.
        nr_neurons_average (int): Number of neurons over which the average is taken.
        subtract_mean (bool): If true, the mean value of the signal is subtracted before FFT. Default is False

    Returns:
        freq, mean_ps, all_ps_dict, mean_firing_rate, mean_firing_freqs_per_neuron
    """

    assert isinstance(spike_monitor, b2.SpikeMonitor), \
        "spike_monitor is not of type SpikeMonitor"

    spiketrains = spike_monitor.spike_trains()
    nr_neurons = len(spiketrains)

    sample_neurons = []
    nr_samples = 0
    if nr_neurons <= nr_neurons_average:
        sample_neurons = range(nr_neurons)
        nr_samples = nr_neurons
    else:
        idxs = np.arange(nr_neurons)
        np.random.shuffle(idxs)
        sample_neurons = idxs[:(nr_neurons_average)]
        nr_samples = nr_neurons_average

    sptrs = filter_spike_trains(spike_monitor.spike_trains(), window_t_min, window_t_max, sample_neurons)
    time_window_size = window_t_max - window_t_min
    discretization_dt = 1./sampling_frequency
    if window_t_max is None:
        window_t_max = max(spike_monitor.t)
    vector_length = 1+int(math.ceil(time_window_size/discretization_dt))  # +1: space for rounding issues
    freq = 0
    spike_count = 0
    all_ps = np.zeros([nr_samples, vector_length/2], float)
    all_ps_dict = dict()
    mean_firing_freqs_per_neuron = dict()
    for i in range(nr_samples):
        idx = sample_neurons[i]
        vec = _spike_train_2_binary_vector(
            sptrs[idx]-window_t_min, vector_length, discretization_dt=discretization_dt)
        ps, freq = _get_spike_train_power_spectrum(vec, discretization_dt, subtract_mean)
        all_ps[i, :] = ps
        all_ps_dict[idx] = ps
        nr_spikes = len(sptrs[idx])
        nu_avg = nr_spikes / time_window_size
        # print(nu_avg)
        mean_firing_freqs_per_neuron[idx] = nu_avg
        spike_count += nr_spikes  # count in the subsample which is filtered to [window_t_min, window_t_max]

    mean_ps = np.mean(all_ps, 0)
    mean_firing_rate = spike_count / nr_samples / time_window_size
    print("mean_firing_rate:{}".format(mean_firing_rate))
    return freq, mean_ps, all_ps_dict, mean_firing_rate, mean_firing_freqs_per_neuron


def get_population_activity_power_spectrum(
        rate_monitor, delta_f, k_repetitions, T_init=100*b2.ms, subtract_mean_activity=False):
    """
    Computes the power spectrum of the population activity A(t) (=rate_monitor.rate)

    Args:
        rate_monitor (RateMonitor): Brian2 rate monitor. rate_monitor.rate is the signal being
            analysed here. The temporal resolution is read from rate_monitor.clock.dt
        delta_f (Quantity): The desired frequency resolution.
        k_repetitions (int): The data rate_monitor.rate is split into k_repetitions which are FFT'd
            independently and then averaged in frequency domain.
        T_init (Quantity): Rates in the time interval [0, T_init] are removed before doing the
            Fourier transform. Use this parameter to ignore the initial transient signals of the simulation.
        subtract_mean_activity (bool): If true, the mean value of the signal is subtracted. Default is False

    Returns:
        freqs, ps, average_population_rate
    """
    data = rate_monitor.rate/b2.Hz
    delta_t = rate_monitor.clock.dt
    f_max = 1./(2. * delta_t)
    N_signal = int(2 * f_max / delta_f)
    T_signal = N_signal * delta_t
    N_init = int(T_init/delta_t)
    N_required = k_repetitions * N_signal + N_init
    N_data = len(data)

    # print("N_data={}, N_required={}".format(N_data,N_required))
    if (N_data < N_required):
        err_msg = "Inconsistent parameters. k_repetitions require {} samples." \
                  " rate_monitor.rate contains {} samples.".format(N_required, N_data)
        raise ValueError(err_msg)
    if N_data > N_required:
        # print("drop samples")
        data = data[:N_required]
    # print("length after dropping end:{}".format(len(data)))
    data = data[N_init:]
    # print("length after dropping init:{}".format(len(data)))
    average_population_rate = np.mean(data)
    if subtract_mean_activity:
        data = data - average_population_rate
    average_population_rate *= b2.Hz
    data = data.reshape(k_repetitions, N_signal)  # reshape into one row per repetition (k)
    k_ps = np.abs(np.fft.fft(data))**2
    ps = np.mean(k_ps, 0)
    # normalize
    ps = ps * delta_t / N_signal  # TODO: verify: subtract 1 (N_signal-1)?
    freqs = np.fft.fftfreq(N_signal, delta_t)
    ps = ps[:(N_signal/2)]
    freqs = freqs[:(N_signal/2)]
    return freqs, ps, average_population_rate
