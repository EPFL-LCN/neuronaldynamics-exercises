from neurodynex3.adex_model import AdEx
from neurodynex3.tools import input_factory
import brian2 as b2


def test_simulate_exponential_IF_neuron():
    """Test if simulates simulate_AdEx_neuron generates two spikes"""
    current = input_factory.get_step_current(0, 0, 1. * b2.ms, 0.5 * b2.nA)
    state_monitor, spike_monitor = AdEx.simulate_AdEx_neuron(I_stim=current, simulation_time=1.5 * b2.ms)
    nr_spikes = spike_monitor.count[0]
    # print("nr_spikes:{}".format(nr_spikes))
    assert nr_spikes == 2, \
        "simulation error: Pulse current did not trigger exactly two spikes. " \
        "Check if the AdEx default values did change."
