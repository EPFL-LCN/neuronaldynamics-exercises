def test_filter_spike_trains():
    """Test filtering of spike train dict"""
    import neurodynex3.tools.spike_tools as spike_tools
    from brian2 import ms
    from brian2 import ms
    st0 = [0, 1, 2] * ms
    st1 = [10, 11] * ms
    st2 = [2, 3, 4, 5] * ms
    spike_trains = dict()
    spike_trains[0] = st0
    spike_trains[1] = st1
    spike_trains[2] = st2
    fst = spike_tools.filter_spike_trains(spike_trains)
    assert len(fst) == 3, "filtered spike trains does not have same number of neurons as spike_trains (a)"
    assert ((fst[0]) == st0).all(), "Default filter does not preserve data (a)"
    assert ((fst[1]) == st1).all(), "Default filter does not preserve data (b)"
    assert ((fst[2]) == st2).all(), "Default filter does not preserve data (c)"
    fst = spike_tools.filter_spike_trains(spike_trains, 2*ms)
    assert len(fst) == 3, "filtered spike trains does not have same number of neurons as spike_trains (b)"
    assert (fst[0]) == [2]*ms, "unexpected result from filter_spike_trains(spike_trains, 2*ms) (1) (a)"
    assert ((fst[1]) == st1).all(), "unexpected result from filter_spike_trains(spike_trains, 2*ms) (b)"
    assert ((fst[2]) == st2).all(), "unexpected result from filter_spike_trains(spike_trains, 2*ms) (c)"
    fst = spike_tools.filter_spike_trains(spike_trains, 2.999999*ms, 5*ms)
    assert len(fst) == 3, "filtered spike trains does not have same number of neurons as spike_trains (c)"
    assert (len(fst[0]) == 0), "1unexpected result from filter_spike_trains(spike_trains, 2.999999*ms, 5*ms) (a)"
    assert (len(fst[1]) == 0), "2unexpected result from filter_spike_trains(spike_trains, 2.999999*ms, 5*ms) (b)"
    assert ((fst[2]) == [3, 4]*ms).all(), \
        "3unexpected result from filter_spike_trains(spike_trains, 2.999999*ms, 5*ms) (c)"
