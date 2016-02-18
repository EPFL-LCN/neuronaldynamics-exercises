from neurodynex.leaky_integrate_and_fire.LIF import LIF_Sinus, LIF_Step


def test_runnable_Step():
    """Test if LIF_Step is runnable."""
    LIF_Step()


def test_runnable_Sinus():
    """Test if LIF_Sinus is runnable."""
    LIF_Sinus()
