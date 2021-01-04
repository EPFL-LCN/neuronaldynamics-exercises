# import matplotlib
# matplotlib.use("Agg")  # needed for plotting on travis


def test_pattern_factory():
    """ Test hopfield_network.pattern_tools """
    import neurodynex3.hopfield_network.pattern_tools as tools
    pattern_size = 6
    factory = tools.PatternFactory(pattern_size)
    p1 = factory.create_checkerboard()
    assert len(p1) == pattern_size


def test_overlap():
    """ Test hopfield_network.pattern_tools overlap"""
    import neurodynex3.hopfield_network.pattern_tools as tools
    pattern_size = 10
    factory = tools.PatternFactory(pattern_size)
    p1 = factory.create_checkerboard()
    p2 = factory.create_all_on()
    overlap = tools.compute_overlap(p1, p2)
    assert overlap == 0.0  # works for checkerboards with even valued size


def test_load_alphabet():
    """Test if the alphabet patterns can be loaded"""
    import neurodynex3.hopfield_network.pattern_tools as pattern_tools
    abc_dictionary = pattern_tools.load_alphabet()
    assert 'A' in abc_dictionary, \
        "Alphabet dict not correctly loaded. Key not accessible"
    assert abc_dictionary['A'].shape == (10, 10), \
        "Letter is not of shape (10,10)"
