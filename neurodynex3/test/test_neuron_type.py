def test_neurons_type():
    """Test if NeuronX and NeuronY constructors are callable"""
    from neurodynex3.neuron_type import neurons
    a_neuron_of_type_X = neurons.NeuronX()
    a_neuron_of_type_Y = neurons.NeuronY()
    assert a_neuron_of_type_X is not None, "Constructor NeuronX did not return an instance"
    assert a_neuron_of_type_Y is not None, "Constructor NeuronY did not return an instance"


def test_neurons_run():
    """Test if neuron functions are runnable."""
    import brian2 as b2
    from neurodynex3.tools import input_factory
    from neurodynex3.neuron_type import neurons
    # create an input current

    input_current = input_factory.get_step_current(1, 2, 1. * b2.ms, 0.1 * b2.pA)

    # get an instance of class NeuronX
    a_neuron_of_type_X = neurons.NeuronX()  # we do not know if it's type I or II
    state_monitor = a_neuron_of_type_X.run(input_current, 2 * b2.ms)
    assert isinstance(state_monitor, b2.StateMonitor), "a_neuron_of_type_X.run did not return a StateMonitor"

    a_neuron_of_type_Y = neurons.NeuronY()  # we do not know if it's type I or II
    state_monitor = a_neuron_of_type_Y.run(input_current, 2 * b2.ms)
    assert isinstance(state_monitor, b2.StateMonitor), "a_neuron_of_type_Y.run did not return a StateMonitor"
