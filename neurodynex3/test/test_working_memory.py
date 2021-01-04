from neurodynex3.working_memory_network import wm_model
import brian2 as b2


def test_woking_memory_sim():
    """Test if the working memory circuit is initialized and simulated for 1ms"""
    b2.defaultclock.dt = 0.2 * b2.ms
    wm_model.simulate_wm(N_excitatory=40, N_inhibitory=10, sim_time=1. * b2.ms)
