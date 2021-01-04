import brian2 as b2
from neurodynex3.brunel_model import LIF_spiking_network

print("b2.__version__={}".format(b2.__version__))


def test_LIF_spiking_network():
    """Test LIF spiking network: simulate_brunel_network(short pulse, 1ms, default values)"""
    rate_monitor, spike_monitor, voltage_monitor, idx_monitored_neurons =\
        LIF_spiking_network.simulate_brunel_network(
            N_Excit=10,
            N_Inhib=None,
            N_extern=10*LIF_spiking_network.N_POISSON_INPUT,
            connection_probability=LIF_spiking_network.CONNECTION_PROBABILITY_EPSILON,
            w0=10*LIF_spiking_network.SYNAPTIC_WEIGHT_W0,
            g=LIF_spiking_network.RELATIVE_INHIBITORY_STRENGTH_G,
            synaptic_delay=LIF_spiking_network.SYNAPTIC_DELAY,
            poisson_input_rate=LIF_spiking_network.POISSON_INPUT_RATE,
            v_rest=LIF_spiking_network.V_REST,
            v_reset=LIF_spiking_network.V_RESET,
            firing_threshold=LIF_spiking_network.FIRING_THRESHOLD,
            membrane_time_scale=LIF_spiking_network.MEMBRANE_TIME_SCALE,
            abs_refractory_period=LIF_spiking_network.ABSOLUTE_REFRACTORY_PERIOD,
            monitored_subset_size=1,
            sim_time=10.*b2.ms)
    nr_spikes = spike_monitor.num_spikes
    # print("nr_spikes:{}".format(nr_spikes))
    assert nr_spikes > 0, \
        "simulation error: Brunel Network did not spike. Check if the LIF default values did change."
    assert isinstance(rate_monitor, b2.PopulationRateMonitor),\
        "first return value is not of type PopulationRateMonitor"
    assert isinstance(spike_monitor, b2.SpikeMonitor), \
        "second return value is not of type SpikeMonitor"
    assert (voltage_monitor is None) or (isinstance(voltage_monitor, b2.StateMonitor)), \
        "third return value is not of type StateMonitor"
