import matplotlib
matplotlib.use('Agg')  # needed for plotting on travis


def test_runnable_Step():
    """Test if HH_Step is runnable."""
    from neurodynex.hodgkin_huxley.HH import HH_Step
    HH_Step(tend=10)


def test_runnable_Sinus():
    """Test if HH_Sinus is runnable."""
    from neurodynex.hodgkin_huxley.HH import HH_Sinus
    HH_Sinus(tend=10)


def test_runnable_Ramp():
    """Test if HH_Ramp is runnable."""
    from neurodynex.hodgkin_huxley.HH import HH_Ramp
    HH_Ramp(tend=10)
