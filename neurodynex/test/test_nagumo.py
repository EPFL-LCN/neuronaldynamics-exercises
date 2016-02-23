import matplotlib
matplotlib.use('Agg')  # needed for plotting on travis


def test_runnable_get_trajectory():
    """Test if get_trajectory is runnable."""
    from neurodynex.phase_plane_analysis.fitzhugh_nagumo import get_trajectory
    get_trajectory()


def test_runnable_get_fixed_point():
    """Test if get_fixed_point is runnable."""
    from neurodynex.phase_plane_analysis.fitzhugh_nagumo import get_fixed_point
    get_fixed_point()


def test_runnable_plot_flow():
    """Test if plot_flow is runnable."""
    from neurodynex.phase_plane_analysis.fitzhugh_nagumo import plot_flow
    plot_flow()
