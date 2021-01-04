from neurodynex3.exponential_integrate_fire import exp_IF
from neurodynex3.tools import input_factory
import brian2 as b2


def test_simulate_exponential_IF_neuron():
    """Test exponential-integrate-and-fire model"""
    c = input_factory.get_step_current(0, 2, 1 * b2.ms, amplitude=8.5 * b2.namp)
    m, spike_monitor = exp_IF.simulate_exponential_IF_neuron(I_stim=c, simulation_time=2.5 * b2.ms)
    nr_spikes = spike_monitor.count[0]
    # print("nr_spikes:{}".format(nr_spikes))
    assert nr_spikes == 1, \
        "simulation error: Pulse current did not trigger exactly one spike. " \
        "Check if the exp-IF default values did change."
