import matplotlib
matplotlib.use('Agg')  # needed for plotting on travis


def run_neuron(c):
    n = c()
    n.step(t_end=10)
    n.get_rate(1., t_end=10)


def test_neurons():
    """Test if neuron functions are runnable."""
    from neurodynex.neuron_type.neurons import NeuronTypeOne, NeuronTypeTwo
    for n in [NeuronTypeOne, NeuronTypeTwo]:
        print "Test if neuron %s is runnable." % n
        run_neuron(n)


def test_class_assignment():
    """Test if NeuronX and NeuronY are properly assigned to NeuronTypeOne
    and NeuronTypeTwo."""
    from neurodynex.neuron_type.neurons import NeuronTypeOne, NeuronTypeTwo
    from neurodynex.neuron_type.typeXY import NeuronX, NeuronY
    import numpy as np

    types = np.array([
        NeuronX.get_neuron_type(),
        NeuronY.get_neuron_type()
    ])

    # assert we do exactly one assignment each
    assert sum((NeuronTypeOne == types).tolist()) == 1
    assert sum((NeuronTypeTwo == types).tolist()) == 1
