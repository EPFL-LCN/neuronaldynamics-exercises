"""
Implementation of a decision making model of
[1] Wang, Xiao-Jing. "Probabilistic decision making by slow reverberation in cortical circuits."
Neuron 36.5 (2002): 955-968.

Some parts of this exercise are inspired by material found at
Stanford Univeristy, BIOE 332: Large-Scale Neural Modeling, Kwabena Boahen and Tatiana Engel, 2013

Most parameters do NOT match those found in the original publication!
"""

# This file is part of the exercise code repository accompanying
# the book: Neuronal Dynamics (see http://neuronaldynamics.epfl.ch)
# located at http://github.com/EPFL-LCN/neuronaldynamics-exercises.

# This free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License 2.0 as published by the
# Free Software Foundation. You should have received a copy of the
# GNU General Public License along with the repository. If not,
# see http://www.gnu.org/licenses/.

# Should you reuse and publish the code for your own purposes,
# please cite the book or point to the webpage http://neuronaldynamics.epfl.ch.

# Wulfram Gerstner, Werner M. Kistler, Richard Naud, and Liam Paninski.
# Neuronal Dynamics: From Single Neurons to Networks and Models of Cognition.
# Cambridge University Press, 2014.


import brian2 as b2
from brian2 import NeuronGroup, Synapses, PoissonInput, PoissonGroup, network_operation
from brian2.monitors import StateMonitor, SpikeMonitor, PopulationRateMonitor
from random import sample
from numpy.random import uniform
from neurodynex.tools import plot_tools
import numpy
import matplotlib.pyplot as plt
from math import ceil, floor
import time

b2.defaultclock.dt = 0.10 * b2.ms

def sim_decision_making_network(N_Excit=384, N_Inhib=96, weight_scaling_factor=5.33,
                                t_stimulus_start=100 * b2.ms, t_stimulus_duration=9999 * b2.ms, coherence_level=0.,
                                stimulus_update_interval=25 * b2.ms, nu0_mean_stimulus_Hz=45.,
                                nu0_std_stimulus_Hz = 10.,
                                N_extern=1000, firing_rate_extern=9.5 * b2.Hz,
                                w_pos=2.0,
                                max_sim_time=200. * b2.ms, stop_condition_rate=None,
                                monitored_subset_size=512):

    print("sim start: {}".format(time.ctime()))
    t_stimulus_end = t_stimulus_start + t_stimulus_duration


    f_Subpop_size = 0.25  # .15 in publication [1]
    N_Group_A = int(N_Excit * f_Subpop_size)  # size of the excitatory subpopulation sensitive to stimulus A
    N_Group_B = N_Group_A  # size of the excitatory subpopulation sensitive to stimulus B
    N_Group_Z = N_Excit - N_Group_A - N_Group_B  # (1-2f)Ne excitatory neurons do not respond to either stimulus.

    Cm_excit = 0.5 * b2.nF  # membrane capacitance of excitatory neurons
    G_leak_excit = 25.0 * b2.nS  # leak conductance
    E_leak_excit = -70.0 * b2.mV  # reversal potential
    v_spike_thr_excit = -50.0 * b2.mV  # spike condition
    v_reset_excit = -60.0 * b2.mV  # reset voltage after spike
    t_abs_refract_excit = 2. * b2.ms  # absolute refractory period

    # specify the inhibitory interneurons:
    # N_Inhib = 200
    Cm_inhib = 0.2 * b2.nF
    G_leak_inhib = 20.0 * b2.nS
    E_leak_inhib = -70.0 * b2.mV
    v_spike_thr_inhib = -50.0 * b2.mV
    v_reset_inhib = -60.0 * b2.mV
    t_abs_refract_inhib = 1.0 * b2.ms

    # specify the AMPA synapses
    E_AMPA = 0.0 * b2.mV
    tau_AMPA = 2.5 * b2.ms

    # specify the GABA synapses
    E_GABA = -70.0 * b2.mV
    tau_GABA = 5.0 * b2.ms

    # specify the NMDA synapses
    E_NMDA = 0.0 * b2.mV
    tau_NMDA_s = 100.0 * b2.ms
    tau_NMDA_x = 2. * b2.ms
    alpha_NMDA = 0.5 * b2.kHz

    # projections from the external population
    g_AMPA_extern2inhib = 1.62 * b2.nS
    g_AMPA_extern2excit = 2.1 * b2.nS

    # projectsions from the inhibitory populations
    g_GABA_inhib2inhib = weight_scaling_factor * 1.25 * b2.nS
    g_GABA_inhib2excit = weight_scaling_factor * 1.60 * b2.nS

    # projections from the excitatory population
    g_AMPA_excit2excit = weight_scaling_factor * 0.012 * b2.nS
    g_AMPA_excit2inhib = weight_scaling_factor * 0.015 * b2.nS
    g_NMDA_excit2excit = weight_scaling_factor * 0.040 * b2.nS
    g_NMDA_excit2inhib = weight_scaling_factor * 0.045 * b2.nS

    # weights and "adjusted" weights.
    w_neg = 1. - f_Subpop_size * (w_pos - 1.) / (1. - f_Subpop_size)
    # We use the same postsyn AMPA and NMDA conductances. Adjust the weights coming from different sources:
    w_ext2inhib = g_AMPA_extern2inhib / g_AMPA_excit2inhib
    w_ext2excit = g_AMPA_extern2excit / g_AMPA_excit2excit
    # other weights are 1
    # print("w_neg={}, w_ext2inhib={}, w_ext2excit={}".format(w_neg, w_ext2inhib, w_ext2excit))


    # Define the inhibitory population
    # dynamics:
    inhib_lif_dynamics = """
        s_NMDA_total : 1  # the post synaptic sum of s. compare with s_NMDA_presyn
        dv/dt = (
        - G_leak_inhib * (v-E_leak_inhib)
        - g_AMPA_excit2inhib * s_AMPA * (v-E_AMPA)
        - g_GABA_inhib2inhib * s_GABA * (v-E_GABA)
        - g_NMDA_excit2inhib * s_NMDA_total * (v-E_NMDA)/(1.0+1.0*exp(-0.062*v/volt)/3.57)
        )/Cm_inhib : volt (unless refractory)
        ds_AMPA/dt = -s_AMPA/tau_AMPA : 1
        ds_GABA/dt = -s_GABA/tau_GABA : 1
    """

    inhib_pop = NeuronGroup(
        N_Inhib, model=inhib_lif_dynamics,
        threshold="v>v_spike_thr_inhib", reset="v=v_reset_inhib", refractory=t_abs_refract_inhib,
        method="rk2")
    # initialize with random voltages:
    inhib_pop.v = uniform(v_spike_thr_inhib / b2.mV - 5., high=v_spike_thr_inhib / b2.mV - 2., size=N_Inhib) * b2.mV

    # Specify the excitatory population:
    # dynamics:
    excit_lif_dynamics = """
        s_NMDA_total : 1  # the post synaptic sum of s. compare with s_NMDA_presyn
        dv/dt = (
        - G_leak_excit * (v-E_leak_excit)
        - g_AMPA_excit2excit * s_AMPA * (v-E_AMPA)
        - g_GABA_inhib2excit * s_GABA * (v-E_GABA)
        - g_NMDA_excit2excit * s_NMDA_total * (v-E_NMDA)/(1.0+1.0*exp(-0.062*v/volt)/3.57)
        )/Cm_excit : volt (unless refractory)
        ds_AMPA/dt = -s_AMPA/tau_AMPA : 1
        ds_GABA/dt = -s_GABA/tau_GABA : 1
        ds_NMDA/dt = -s_NMDA/tau_NMDA_s + alpha_NMDA * x * (1-s_NMDA) : 1
        dx/dt = -x/tau_NMDA_x : 1
    """

    # define the three excitatory subpopulations.
    # A: subpop receiving stimulus A
    excit_pop_A = NeuronGroup(N_Group_A, model=excit_lif_dynamics,
                              threshold="v>v_spike_thr_excit", reset="v=v_reset_excit",
                              refractory=t_abs_refract_excit, method="rk2")
    excit_pop_A.v = uniform(E_leak_excit / b2.mV, high=E_leak_excit / b2.mV + 5., size=excit_pop_A.N) * b2.mV

    # B: subpop receiving stimulus B
    excit_pop_B = NeuronGroup(N_Group_B, model=excit_lif_dynamics, threshold="v>v_spike_thr_excit",
                              reset="v=v_reset_excit", refractory=t_abs_refract_excit, method="rk2")
    excit_pop_B.v = uniform(E_leak_excit / b2.mV, high=E_leak_excit / b2.mV + 5., size=excit_pop_B.N) * b2.mV
    # Z: non-sensitive
    excit_pop_Z = NeuronGroup(N_Group_Z, model=excit_lif_dynamics,
                              threshold="v>v_spike_thr_excit", reset="v=v_reset_excit",
                              refractory=t_abs_refract_excit, method="rk2")
    excit_pop_Z.v = uniform(E_leak_excit / b2.mV, high=E_leak_excit / b2.mV + 5., size=excit_pop_Z.N) * b2.mV

    # now define the connections:
    # projections FROM EXTERNAL POISSON GROUP: ####################################################
    poisson2Inhib = PoissonInput(target=inhib_pop, target_var="s_AMPA",
                                 N=N_extern, rate=firing_rate_extern, weight=w_ext2inhib)
    poisson2A = PoissonInput(target=excit_pop_A, target_var="s_AMPA",
                             N=N_extern, rate=firing_rate_extern, weight=w_ext2excit)

    poisson2B = PoissonInput(target=excit_pop_B, target_var="s_AMPA",
                             N=N_extern, rate=firing_rate_extern, weight=w_ext2excit)
    poisson2Z = PoissonInput(target=excit_pop_Z, target_var="s_AMPA",
                             N=N_extern, rate=firing_rate_extern, weight=w_ext2excit)

    ###############################################################################################

    # GABA projections FROM INHIBITORY population: ################################################
    syn_inhib2inhib = Synapses(inhib_pop, target=inhib_pop, on_pre="s_GABA += 1.0", delay=0.5 * b2.ms)
    syn_inhib2inhib.connect(p=1.)
    syn_inhib2A = Synapses(inhib_pop, target=excit_pop_A, on_pre="s_GABA += 1.0", delay=0.5 * b2.ms)
    syn_inhib2A.connect(p=1.)
    syn_inhib2B = Synapses(inhib_pop, target=excit_pop_B, on_pre="s_GABA += 1.0", delay=0.5 * b2.ms)
    syn_inhib2B.connect(p=1.)
    syn_inhib2Z = Synapses(inhib_pop, target=excit_pop_Z, on_pre="s_GABA += 1.0", delay=0.5 * b2.ms)
    syn_inhib2Z.connect(p=1.)
    ###############################################################################################

    # AMPA projections FROM EXCITATORY A: #########################################################
    syn_AMPA_A2A = Synapses(excit_pop_A, target=excit_pop_A, on_pre="s_AMPA += w_pos", delay=0.5 * b2.ms)
    syn_AMPA_A2A.connect(p=1.)
    syn_AMPA_A2B = Synapses(excit_pop_A, target=excit_pop_B, on_pre="s_AMPA += w_neg", delay=0.5 * b2.ms)
    syn_AMPA_A2B.connect(p=1.)
    syn_AMPA_A2Z = Synapses(excit_pop_A, target=excit_pop_Z, on_pre="s_AMPA += 1.0", delay=0.5 * b2.ms)
    syn_AMPA_A2Z.connect(p=1.)
    syn_AMPA_A2inhib = Synapses(excit_pop_A, target=inhib_pop, on_pre="s_AMPA += 1.0", delay=0.5 * b2.ms)
    syn_AMPA_A2inhib.connect(p=1.)
    ###############################################################################################

    # AMPA projections FROM EXCITATORY B: #########################################################
    syn_AMPA_B2A = Synapses(excit_pop_B, target=excit_pop_A, on_pre="s_AMPA += w_neg", delay=0.5 * b2.ms)
    syn_AMPA_B2A.connect(p=1.)
    syn_AMPA_B2B = Synapses(excit_pop_B, target=excit_pop_B, on_pre="s_AMPA += w_pos", delay=0.5 * b2.ms)
    syn_AMPA_B2B.connect(p=1.)
    syn_AMPA_B2Z = Synapses(excit_pop_B, target=excit_pop_Z, on_pre="s_AMPA += 1.0", delay=0.5 * b2.ms)
    syn_AMPA_B2Z.connect(p=1.)
    syn_AMPA_B2inhib = Synapses(excit_pop_B, target=inhib_pop, on_pre="s_AMPA += 1.0", delay=0.5 * b2.ms)
    syn_AMPA_B2inhib.connect(p=1.)
    ###############################################################################################

    # AMPA projections FROM EXCITATORY Z: #########################################################
    syn_AMPA_Z2A = Synapses(excit_pop_Z, target=excit_pop_A, on_pre="s_AMPA += 1.0", delay=0.5 * b2.ms)
    syn_AMPA_Z2A.connect(p=1.)
    syn_AMPA_Z2B = Synapses(excit_pop_Z, target=excit_pop_B, on_pre="s_AMPA += 1.0", delay=0.5 * b2.ms)
    syn_AMPA_Z2B.connect(p=1.)
    syn_AMPA_Z2Z = Synapses(excit_pop_Z, target=excit_pop_Z, on_pre="s_AMPA += 1.0", delay=0.5 * b2.ms)
    syn_AMPA_Z2Z.connect(p=1.)
    syn_AMPA_Z2inhib = Synapses(excit_pop_Z, target=inhib_pop, on_pre="s_AMPA += 1.0", delay=0.5 * b2.ms)
    syn_AMPA_Z2inhib.connect(p=1.)
    ###############################################################################################

    # NMDA projections FROM EXCITATORY to INHIB, A,B,Z
    @network_operation()
    def update_nmda_sum():
        sum_sNMDA_A = sum(excit_pop_A.s_NMDA)
        sum_sNMDA_B = sum(excit_pop_B.s_NMDA)
        sum_sNMDA_Z = sum(excit_pop_Z.s_NMDA)
        inhib_pop.s_NMDA_total = (1.0 * sum_sNMDA_A + 1.0 * sum_sNMDA_B + 1.0 * sum_sNMDA_Z)
        excit_pop_A.s_NMDA_total = (w_pos * sum_sNMDA_A + w_neg * sum_sNMDA_B + w_neg * sum_sNMDA_Z)
        excit_pop_B.s_NMDA_total = (w_neg * sum_sNMDA_A + w_pos * sum_sNMDA_B + w_neg * sum_sNMDA_Z)
        excit_pop_Z.s_NMDA_total = (1.0 * sum_sNMDA_A + 1.0 * sum_sNMDA_B + 1.0 * sum_sNMDA_Z)

    # set a self-recurrent synapse to introduce a delay when updating the intermediate
    # gating variable x
    syn_x_A2A = Synapses(excit_pop_A, excit_pop_A, on_pre="x += 1.", delay=0.5 * b2.ms)
    syn_x_A2A.connect(j="i")
    syn_x_B2B = Synapses(excit_pop_B, excit_pop_B, on_pre="x += 1.", delay=0.5 * b2.ms)
    syn_x_B2B.connect(j="i")
    syn_x_Z2Z = Synapses(excit_pop_Z, excit_pop_Z, on_pre="x += 1.", delay=0.5 * b2.ms)
    syn_x_Z2Z.connect(j="i")
    ###############################################################################################

    # Define the stimulus: two PoissonInput with time time-dependent mean.
    poissonStimulus2A = PoissonGroup(N_Group_A, 0. * b2.Hz)
    syn_Stim2A = Synapses(poissonStimulus2A, excit_pop_A, on_pre="s_AMPA+=w_ext2excit")
    syn_Stim2A.connect(j="i")
    poissonStimulus2B = PoissonGroup(N_Group_B, 0. * b2.Hz)
    syn_Stim2B = Synapses(poissonStimulus2B, excit_pop_B, on_pre="s_AMPA+=w_ext2excit")
    syn_Stim2B.connect(j="i")

    @network_operation(dt=stimulus_update_interval)
    def update_poisson_stimulus(t):
        if t >= t_stimulus_start and t < t_stimulus_end:
            offset_A = nu0_mean_stimulus_Hz * (0.5 + 0.5 * coherence_level)
            offset_B = nu0_mean_stimulus_Hz * (0.5 - 0.5 * coherence_level)

            rate_A = numpy.random.normal(offset_A, nu0_std_stimulus_Hz)
            rate_A = (max(0, rate_A)) * b2.Hz  # avoid negative rate
            rate_B = numpy.random.normal(offset_B, nu0_std_stimulus_Hz)
            rate_B = (max(0, rate_B)) * b2.Hz

            poissonStimulus2A.rates = rate_A
            poissonStimulus2B.rates = rate_B
            # print("stim on. rate_A= {}, rate_B = {}".format(rate_A, rate_B))
        else:
            # print("stim off")
            poissonStimulus2A.rates = 0.
            poissonStimulus2B.rates = 0.


    ###############################################################################################


    def get_monitors(pop, monitored_subset_size):
        monitored_subset_size = min(monitored_subset_size, pop.N)
        idx_monitored_neurons = sample(range(pop.N), monitored_subset_size)
        rate_monitor = PopulationRateMonitor(pop)
        # record= some_list is not supported? :-(
        spike_monitor = SpikeMonitor(pop, record=idx_monitored_neurons)
        voltage_monitor = StateMonitor(pop, "v", record=idx_monitored_neurons)
        return rate_monitor, spike_monitor, voltage_monitor, idx_monitored_neurons

    # collect data of a subset of neurons:
    rate_monitor_inhib, spike_monitor_inhib, voltage_monitor_inhib, idx_monitored_neurons_inhib = \
        get_monitors(inhib_pop, monitored_subset_size)

    rate_monitor_A, spike_monitor_A, voltage_monitor_A, idx_monitored_neurons_A = \
        get_monitors(excit_pop_A, monitored_subset_size)

    rate_monitor_B, spike_monitor_B, voltage_monitor_B, idx_monitored_neurons_B = \
        get_monitors(excit_pop_B, monitored_subset_size)

    rate_monitor_Z, spike_monitor_Z, voltage_monitor_Z, idx_monitored_neurons_Z = \
        get_monitors(excit_pop_Z, monitored_subset_size)

    if stop_condition_rate is None:
        b2.run(max_sim_time)
    else:
        sim_sum = 0. * b2.ms
        sim_batch = 100. * b2.ms
        samples_in_batch = int(floor(sim_batch / b2.defaultclock.dt))
        avg_rate_in_batch = 0
        while (sim_sum < max_sim_time) and (avg_rate_in_batch < stop_condition_rate):
            b2.run(sim_batch)
            avg_A = numpy.mean(rate_monitor_A.rate[-samples_in_batch:])
            avg_B = numpy.mean(rate_monitor_B.rate[-samples_in_batch:])
            avg_rate_in_batch = max(avg_A, avg_B)
            sim_sum += sim_batch

    print("sim end: {}".format(time.ctime()))
    ret_vals = dict()

    ret_vals["rate_monitor_A"] = rate_monitor_A
    ret_vals["spike_monitor_A"] = spike_monitor_A
    ret_vals["voltage_monitor_A"] = voltage_monitor_A
    ret_vals["idx_monitored_neurons_A"] = idx_monitored_neurons_A

    ret_vals["rate_monitor_B"] = rate_monitor_B
    ret_vals["spike_monitor_B"] = spike_monitor_B
    ret_vals["voltage_monitor_B"] = voltage_monitor_B
    ret_vals["idx_monitored_neurons_B"] = idx_monitored_neurons_B

    ret_vals["rate_monitor_Z"] = rate_monitor_Z
    ret_vals["spike_monitor_Z"] = spike_monitor_Z
    ret_vals["voltage_monitor_Z"] = voltage_monitor_Z
    ret_vals["idx_monitored_neurons_Z"] = idx_monitored_neurons_Z

    ret_vals["rate_monitor_inhib"] = rate_monitor_inhib
    ret_vals["spike_monitor_inhib"] = spike_monitor_inhib
    ret_vals["voltage_monitor_inhib"] = voltage_monitor_inhib
    ret_vals["idx_monitored_neurons_inhib"] = idx_monitored_neurons_inhib

    return ret_vals


def run_multiple_simulations(
        f_get_decision_time, coherence_levels, nr_repetitions,
        max_sim_time=123 * b2.ms, rate_threshold=5 * b2.Hz, avg_window_width=5 * b2.ms,
        N_excit=384, N_inhib=96, weight_scaling=5.33,
        t_stim_start=100 * b2.ms, t_stim_duration=9999 * b2.ms,
        nu0_mean_stim_Hz=50., nu0_std_stim_Hz = 12., stim_upd_interval=20*b2.ms):

    nr_coherence = len(coherence_levels)
    count_A = numpy.zeros(nr_coherence, dtype=numpy.int8)
    count_B = numpy.zeros(nr_coherence, dtype=numpy.int8)
    count_No = numpy.zeros(nr_coherence, dtype=numpy.int8)

    time_to_A = numpy.zeros((nr_coherence, nr_repetitions))
    time_to_B = numpy.zeros((nr_coherence, nr_repetitions))

    for i_coherence in range(nr_coherence):
        c = coherence_levels[i_coherence]
        print("********************************************")
        print("coherence_level={}".format(c))
        for i_run in range(nr_repetitions):
            print("i_run={}".format(i_run))
            results = sim_decision_making_network(
                N_Excit=N_excit, N_Inhib=N_inhib, weight_scaling_factor=weight_scaling,
                t_stimulus_start=t_stim_start, t_stimulus_duration=t_stim_duration, coherence_level=c,
                max_sim_time=max_sim_time, stop_condition_rate=rate_threshold,
                nu0_mean_stimulus_Hz=nu0_mean_stim_Hz, nu0_std_stimulus_Hz=nu0_std_stim_Hz,
                stimulus_update_interval=stim_upd_interval
            )
            t_A, t_B = f_get_decision_time(results["rate_monitor_A"],
                                           results["rate_monitor_B"],
                                           avg_window_width, rate_threshold)
            time_to_A[i_coherence, i_run] = t_A
            time_to_B[i_coherence, i_run] = t_B
            print("t_A={}, t_B={}".format(t_A, t_B))
            if t_A > 0.:
                print("decision: A")
                count_A += 1
            elif t_B > 0.:
                print("decision: B")
                count_B += 1
            else:
                print("no decision")
                count_No += 1
    return time_to_A, time_to_B, count_A, count_B, count_No


def getting_started():
    results = sim_decision_making_network(N_Excit=384, N_Inhib=96, weight_scaling_factor=5.33,
                                          t_stimulus_start= 100. * b2.ms, t_stimulus_duration=1000 * b2.ms,
                                          coherence_level=-0.3,
                                          max_sim_time=800. * b2.ms)
    plot_tools.plot_network_activity(results["rate_monitor_A"], results["spike_monitor_A"],
                                     results["voltage_monitor_A"], t_min=0. * b2.ms, avg_window_width=20. * b2.ms,
                                     sup_title="Left")
    plot_tools.plot_network_activity(results["rate_monitor_B"], results["spike_monitor_B"],
                                     results["voltage_monitor_B"], t_min=0. * b2.ms, avg_window_width=20. * b2.ms,
                                     sup_title="Right")

    plot_tools.plot_network_activity(results["rate_monitor_Z"], results["spike_monitor_Z"],
                                     results["voltage_monitor_Z"], t_min=0. * b2.ms, avg_window_width=20. * b2.ms,
                                     sup_title="Neutral excitatory sub-population")

    plot_tools.plot_network_activity(results["rate_monitor_inhib"], results["spike_monitor_inhib"],
                                     results["voltage_monitor_inhib"],
                                     t_min=0. * b2.ms, avg_window_width=20. * b2.ms,
                                     sup_title="Inhib")

    plt.show()

    # plot_rates(results["rate_monitor_A"], results["rate_monitor_B"])
    # plt.show()


if __name__ == "__main__":
    getting_started()
