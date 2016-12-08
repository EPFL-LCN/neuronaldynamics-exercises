
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


def get_spike_time(voltage_monitor, spike_threshold):
    """
    Detects the spike times in the voltage. The spike time is the value in voltage_monitor.t for
    which voltage_monitor.v[idx] is above threshold AND voltage_monitor.v[idx-1] is below threshold
    (crossing from below).
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
    Wraps a few spike-train related properties into a class.
    """
    def __init__(self, nr_neurons, nr_spikes, all_ISI):
        """

        Args:
            nr_neurons:
            nr_spikes:
            mean_isi:
            std_isi:
            all_ISI: list of ISI values (can be used to plot a histrogram)

        Returns:
            An instance of PopulationSpikeStats
        """
        self._nr_neurons = nr_neurons
        self._nr_spikes = nr_spikes
        self._all_ISI = all_ISI

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
        Coefficient of variation
        """
        cv = np.nan  # init with nan
        if self.mean_isi > 0.:
            cv = self.std_isi / self.mean_isi
        return cv


def get_spike_train_stats(spike_monitor):
    """
    analyses the spike monitor and returns a PopulationSpikeStats instance.
    Args:
        spike_monitor (SpikeMonitor): Brian2 spike monitor

    Returns:
        PopulationSpikeStats
    """
    assert isinstance(spike_monitor, b2.SpikeMonitor), \
        "spike_monitor is not of type SpikeMonitor"
    all_spike_trains = spike_monitor.spike_trains()
    nr_neurons = len(all_spike_trains)
    all_ISI = []
    for i in range(nr_neurons):
        spike_times = all_spike_trains[i]/b2.ms
        nr_spikes = len(spike_times)
        if nr_spikes >= 2:
            isi = np.diff(spike_times)
            all_ISI = np.hstack([all_ISI, isi])
    all_ISI = all_ISI*b2.ms
    stats = PopulationSpikeStats(nr_neurons, spike_monitor.num_spikes, all_ISI)
    return stats
