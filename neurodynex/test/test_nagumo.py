def test_runnable_get_trajectory():
    """Test if get_trajectory is runnable."""
    from neurodynex.phase_plane_analysis.fitzhugh_nagumo import get_trajectory
    get_trajectory(v0=1., w0=2., I=3., eps=0.4, a=5.0, tend=1.234)


def test_runnable_get_fixed_point():
    """Test if get_fixed_point is runnable."""
    from neurodynex.phase_plane_analysis.fitzhugh_nagumo import get_fixed_point
    get_fixed_point()
