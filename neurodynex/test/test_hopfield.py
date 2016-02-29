import matplotlib
matplotlib.use('Agg')  # needed for plotting on travis


def test_alphabet():
    """Test if alphabet is loadable."""
    from neurodynex.hopfield_network.hopfield import\
        load_alphabet
    load_alphabet()


def test_net_random():
    """Test hopfield network with random patterns."""
    from neurodynex.hopfield_network.hopfield import\
        HopfieldNetwork

    size = 8
    n = HopfieldNetwork(size)
    n.make_pattern()
    assert n.patterns.shape == (1, size**2)

    n.run(flip_ratio=.2)
    assert n.overlap(0) == 1.

    n.run(flip_ratio=.8)
    assert n.overlap(0) == -1.


def test_net_alphabet():
    """Test hopfield network with alphabet patterns."""
    from neurodynex.hopfield_network.hopfield import\
        HopfieldNetwork

    size = 10
    n = HopfieldNetwork(size)
    n.make_pattern(letters='lcn')
    assert n.patterns.shape == (3, size**2)

    for i in range(3):
        n.run(mu=i, flip_ratio=.2)
        assert n.overlap(i) == 1.
        n.run(mu=i, flip_ratio=.8)
        assert n.overlap(i) == -1.
