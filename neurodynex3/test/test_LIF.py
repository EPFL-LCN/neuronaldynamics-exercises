from neurodynex3.leaky_integrate_and_fire import LIF
from neurodynex3.tools import input_factory
import brian2 as b2


def test_simulate_LIF_neuron():
    """Test LIF model: simulate_LIF_neuron(short pulse, 1ms, default values)"""
    c = input_factory.get_step_current(0, 9, 0.1 * b2.ms, .02 * b2.uA)
    m, spike_monitor = LIF.simulate_LIF_neuron(c, simulation_time=1.1 * b2.ms)
    nr_spikes = spike_monitor.count[0]
    # print("nr_spikes:{}".format(nr_spikes))
    assert nr_spikes > 0, \
        "simulation error: Pulse current did not trigger a spike. " \
        "Check if the LIF default values did change."
