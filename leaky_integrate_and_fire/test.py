import matplotlib
matplotlib.use('Agg')


def test_runnable_Step():
    """Test if LIF_Step is runnable."""
    from .LIF import LIF_Step
    LIF_Step()


def test_runnable_Sinus():
    """Test if LIF_Step is runnable."""
    from .LIF import LIF_Sinus
    LIF_Sinus()
