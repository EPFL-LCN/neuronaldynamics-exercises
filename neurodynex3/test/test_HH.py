from neurodynex3.hodgkin_huxley import HH
from neurodynex3.tools import input_factory
import brian2 as b2


def test_simulate_HH_neuron():
    """Test Hodgkin-Huxley model: simulate_HH_neuron()"""
    current = input_factory.get_step_current(0, 1, b2.ms, 100. * b2.uA)
    state_monitor = HH.simulate_HH_neuron(current, simulation_time=1. * b2.ms)
    max_voltage = max(state_monitor.vm[0] / b2.mV)
    # print("max_voltage:{}".format(max_voltage))
    assert max_voltage > 50., "simulation error: max voltage is not > 50"
