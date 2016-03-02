import matplotlib
matplotlib.use('Agg')  # needed for plotting on travis


def test_oja():
    """Test if Oja learning rule is runnable."""
    from neurodynex.ojas_rule.oja import run_oja
    run_oja()  # this uses all functions in the module
