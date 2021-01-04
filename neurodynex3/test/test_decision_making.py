from neurodynex3.competing_populations import decision_making
import brian2 as b2


def test_sim_decision_making_network():
    """Test if the decision making circuit is initialized and simulated for 3ms"""
    b2.defaultclock.dt = 0.2 * b2.ms
    results = decision_making\
        .sim_decision_making_network(
            N_Excit=20, N_Inhib=5, weight_scaling_factor=10., t_stimulus_start=0 * b2.ms,
            t_stimulus_duration=9999 * b2.ms, coherence_level=.5, stimulus_update_interval=2 * b2.ms,
            mu0_mean_stimulus_Hz=160., stimulus_std_Hz=20., N_extern=10, firing_rate_extern=9.5 * b2.Hz,
            w_pos=1.70, f_Subpop_size=0.30, max_sim_time=3. * b2.ms,
            stop_condition_rate=None, monitored_subset_size=5)

    assert "rate_monitor_A" in results, \
        "results dict returned from sim_decision_making_network does not contain key rate_monitor_A"
