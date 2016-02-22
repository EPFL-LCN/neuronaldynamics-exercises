import matplotlib
matplotlib.use('Agg')  # needed for plotting on travis


def test_runnable_Step():
    """Test if LIF_Step is runnable."""
    from neurodynex.leaky_integrate_and_fire.LIF import LIF_Step
    LIF_Step(tend=10)


def test_runnable_Sinus():
    """Test if LIF_Sinus is runnable."""
    from neurodynex.leaky_integrate_and_fire.LIF import LIF_Sinus
    LIF_Sinus(tend=10)
