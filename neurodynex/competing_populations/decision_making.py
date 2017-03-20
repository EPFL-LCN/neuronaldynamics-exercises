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
from brian2 import NeuronGroup, Synapses, PoissonInput, network_operation
from brian2.monitors import StateMonitor, SpikeMonitor, PopulationRateMonitor
from random import sample
from numpy.random import uniform
from neurodynex.tools import plot_tools
import numpy
import matplotlib.pyplot as plt
import math
from scipy.special import erf
from numpy.fft import rfft, irfft

b2.defaultclock.dt = 0.02 * b2.ms


def simulate_wm(
        N_extern=1500, firing_rate_extern=6.4 * b2.Hz,
        N_Excit=1600, w_pos=1.7,
        N_Inhib=400,
        sim_time=100. * b2.ms):
    print("sim wm")

    # specify the excitatory pyramidal cells:
    # N_Excit = 800
    f_Subpop_size = 0.15  # as in publication [1]
    N_Group_A = int(N_Excit * f_Subpop_size)  # size of the excitatory subpopulation sensitive to stimulus A
    N_Group_B = N_Group_A  # size of the excitatory subpopulation sensitive to stimulus B
    N_Group_Z = N_Excit - N_Group_A - N_Group_B  # (1-2f)Ne excitatory neurons do not respond to either stimulus.
    Cm_excit = 0.5 * b2.nF  # membrane capacitance of excitatory neurons
    G_leak_excit = 25.0 * b2.nS  # leak conductance
    E_leak_excit = -70.0 * b2.mV  # reversal potential
    v_spike_thr_excit = -50.0 * b2.mV  # spike condition
    v_reset_excit = -57.0 * b2.mV  # reset voltage after spike
    # t_abs_refract_excit = 2.0 * b2.ms  # absolute refractory period
    t_abs_refract_excit = "(1.5 + 1.*rand())*ms"

    # specify the inhibitory interneurons:
    # N_Inhib = 200
    Cm_inhib = 0.2 * b2.nF
    G_leak_inhib = 20.0 * b2.nS
    E_leak_inhib = -70.0 * b2.mV
    v_spike_thr_inhib = -50.0 * b2.mV
    v_reset_inhib = -57.0 * b2.mV
    # t_abs_refract_inhib = 1.0 * b2.ms
    t_abs_refract_inhib = "(.5 + 1.*rand())*ms"

    # specify the AMPA synapses
    E_AMPA = 0.0 * b2.mV
    tau_AMPA = 2.0 * b2.ms

    # specify the GABA synapses
    E_GABA = -70.0 * b2.mV
    tau_GABA = 5.0 * b2.ms

    # specify the NMDA synapses
    E_NMDA = 0.0 * b2.mV
    tau_NMDA_s = 100.0 * b2.ms
    tau_NMDA_x = 2.0 * b2.ms
    alpha_NMDA = 0.5 * b2.kHz

    # projections from the external population
    g_AMPA_extern2inhib = 1.62 * b2.nS
    g_AMPA_extern2excit = 2.1 * b2.nS  # 2.1

    # projectsions from the inhibitory populations
    g_GABA_inhib2inhib = 5.2 * 1.0 * b2.nS
    g_GABA_inhib2excit = 5.2 * 1.3 * b2.nS  # 1.3

    # projections from the excitatory population
    g_AMPA_excit2excit = 0.05 * b2.nS
    g_AMPA_excit2inhib = 1.5 * 0.04 * b2.nS
    g_NMDA_excit2excit = 0.165 * b2.nS
    g_NMDA_excit2inhib = 1.2 * 0.130 * b2.nS  # todo: verify this scaling

    # weights and "adjusted" weights.
    # w_1 = 1.
    # w_pos = 1.6
    w_neg = 1. - f_Subpop_size * (w_pos - 1.) / (1. - f_Subpop_size)
    # We use the same postsyn AMPA and NMDA conductances. Adjust the weights coming from different sources:
    # ... - g_AMPA_extern2inhib * s_AMPA * (v-E_AMPA) ... in the inhib pop
    # ... - g_AMPA_extern2excit * s_AMPA * (v-E_AMPA) ... in the excit pop
    w_ext2inhib = 1. * g_AMPA_extern2inhib / g_AMPA_excit2inhib
    w_ext2excit = 1. * g_AMPA_extern2excit / g_AMPA_excit2excit
    # all other weights are 1
    print("w_neg={}, w_ext2inhib={}, w_ext2excit={}".format(w_neg, w_ext2inhib, w_ext2excit))

    # specify simulation and monitoring
    monitored_subset_size = 200

    ###############################################################################################
    ###############################################################################################

    # define the inhibitory population
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
    inhib_pop.v = uniform(E_leak_inhib / b2.mV, high=v_spike_thr_inhib / b2.mV, size=N_Inhib) * b2.mV

    # specify the excitatory population:
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
    # A: stimulus A
    excit_pop_A = NeuronGroup(N_Group_A, model=excit_lif_dynamics,
                              threshold="v>v_spike_thr_excit", reset="v=v_reset_excit",
                              refractory=t_abs_refract_excit, method="rk2")
    excit_pop_A.v = uniform(E_leak_excit / b2.mV, high=v_spike_thr_excit / b2.mV, size=excit_pop_A.N) * b2.mV

    # B: stimulus B
    excit_pop_B = NeuronGroup(N_Group_B, model=excit_lif_dynamics, threshold="v>v_spike_thr_excit",
                              reset="v=v_reset_excit", refractory=t_abs_refract_excit, method="rk2")
    excit_pop_B.v = uniform(E_leak_excit / b2.mV, high=v_spike_thr_excit / b2.mV, size=excit_pop_B.N) * b2.mV
    # Z: non-sensitive
    excit_pop_Z = NeuronGroup(N_Group_Z, model=excit_lif_dynamics,
                              threshold="v>v_spike_thr_excit", reset="v=v_reset_excit",
                              refractory=t_abs_refract_excit, method="rk2")
    excit_pop_Z.v = uniform(E_leak_excit / b2.mV, high=v_spike_thr_excit / b2.mV, size=excit_pop_Z.N) * b2.mV

    # define the external input: ##################################################################

# now define the connections:
# projections FROM EXTERNAL POISSON GROUP: ####################################################

    poisson2Inhib = PoissonInput(target=inhib_pop, target_var="s_AMPA",
                                 N=N_extern, rate=firing_rate_extern, weight=w_ext2inhib)
    poisson2A = PoissonInput(target=excit_pop_A, target_var="s_AMPA",
                             N=N_extern, rate=firing_rate_extern, weight=w_ext2excit)

    # ToDo: stimulus!
    poisson2B = PoissonInput(target=excit_pop_B, target_var="s_AMPA",
                             N=N_extern, rate=firing_rate_extern, weight=1.05 * w_ext2excit)
    poisson2Z = PoissonInput(target=excit_pop_Z, target_var="s_AMPA",
                             N=N_extern, rate=firing_rate_extern, weight=w_ext2excit)




    # Pg_ext2Ihnib = PoissonGroup(5*N_extern, firing_rate_extern)
    # syn_ext2Inhib = Synapses(Pg_ext2Ihnib, target=inhib_pop, on_pre="s_AMPA += w_ext2inhib")
    # syn_ext2Inhib.connect(p=.1)
    #
    # Pg_ext2A = PoissonGroup(5*N_extern, firing_rate_extern)
    # syn_ext2A = Synapses(Pg_ext2A, target=excit_pop_A, on_pre="s_AMPA += w_ext2excit")
    # syn_ext2A.connect(p=.1)
    #
    # Pg_ext2B = PoissonGroup(5*N_extern, firing_rate_extern)
    # syn_ext2B = Synapses(Pg_ext2B, target=excit_pop_B, on_pre="s_AMPA += w_ext2excit")
    # syn_ext2B.connect(p=.1)
    #
    # Pg_ext2Z = PoissonGroup(5*N_extern, firing_rate_extern)
    # syn_ext2Z = Synapses(Pg_ext2Z, target=excit_pop_Z, on_pre="s_AMPA += w_ext2excit")
    # syn_ext2Z.connect(p=.1)

    # t_stimulus_start = 5. * b2.ms
    # t_stimulus_end = 250. * b2.ms
    # stim_mean = 0.05
    # # stim_sigma =
    # coherence = 1.
    # # @network_operation(dt=10 * b2.ms)
    # def stimulate_network(t):
    #     print("stimulate_network nw op")
    #     if t >= t_stimulus_start and t < t_stimulus_end:
    #         Pg_ext2A.rates_ = firing_rate_extern + stim_mean * (1.0 + coherence) * b2.Hz  # + math.rand()* sigmaMu
    #         Pg_ext2B.rates_ = firing_rate_extern + stim_mean * (1.0 - coherence) * b2.Hz  # + math.rand() * sigmaMu
    #     else:
    #         Pg_ext2A.rates_ = firing_rate_extern
    #         Pg_ext2B.rates_ = firing_rate_extern

    ###############################################################################################

    # now define the connections:

    # GABA projections FROM INHIBITORY population: ################################################
    syn_inhib2inhib = Synapses(inhib_pop, target=inhib_pop, on_pre="s_GABA += 1.0", delay=0.5 * b2.ms)
    syn_inhib2inhib.connect(p=.45)
    syn_inhib2A = Synapses(inhib_pop, target=excit_pop_A, on_pre="s_GABA += 1.0", delay=0.5 * b2.ms)
    syn_inhib2A.connect(p=.45)
    syn_inhib2B = Synapses(inhib_pop, target=excit_pop_B, on_pre="s_GABA += 1.0", delay=0.5 * b2.ms)
    syn_inhib2B.connect(p=.45)
    syn_inhib2Z = Synapses(inhib_pop, target=excit_pop_Z, on_pre="s_GABA += 1.0", delay=0.5 * b2.ms)
    syn_inhib2Z.connect(p=.45)
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
        # underscore switches off unit checking.
        inhib_pop.s_NMDA_total_ = (1.0 * sum_sNMDA_A + 1.0 * sum_sNMDA_B + 1.0 * sum_sNMDA_Z)
        excit_pop_A.s_NMDA_total_ = (w_pos * sum_sNMDA_A + w_neg * sum_sNMDA_B + w_neg * sum_sNMDA_Z)
        excit_pop_B.s_NMDA_total_ = (w_neg * sum_sNMDA_A + w_pos * sum_sNMDA_B + w_neg * sum_sNMDA_Z)
        excit_pop_Z.s_NMDA_total_ = (1.0 * sum_sNMDA_A + 1.0 * sum_sNMDA_B + 1.0 * sum_sNMDA_Z)

    # set a self-recurrent synapse to introduce a delay when updating the intermediate
    # gating variable x
    syn_x_A2A = Synapses(excit_pop_A, excit_pop_A, on_pre="x += 1.", delay=0.5 * b2.ms)
    syn_x_A2A.connect(j="i")
    syn_x_B2B = Synapses(excit_pop_B, excit_pop_B, on_pre="x += 1.", delay=0.5 * b2.ms)
    syn_x_B2B.connect(j="i")
    syn_x_Z2Z = Synapses(excit_pop_Z, excit_pop_Z, on_pre="x += 1.", delay=0.5 * b2.ms)
    syn_x_Z2Z.connect(j="i")
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

    b2.run(sim_time)

    fig, ax_raster, ax_rate, ax_voltage = plot_tools.plot_network_activity(
        rate_monitor_A, spike_monitor_A, voltage_monitor_A, t_min=0. * b2.ms, window_width=40. * b2.ms)
    fig.canvas.set_window_title('Population A')

    fig, ax_raster, ax_rate, ax_voltage = plot_tools.plot_network_activity(
        rate_monitor_B, spike_monitor_B, voltage_monitor_B, t_min=0. * b2.ms, window_width=40. * b2.ms)
    fig.canvas.set_window_title('Population B')

    fig, ax_raster, ax_rate, ax_voltage = plot_tools.plot_network_activity(
        rate_monitor_Z, spike_monitor_Z, voltage_monitor_Z, t_min=0. * b2.ms)
    fig.canvas.set_window_title('Population Z')

    fig, ax_raster, ax_rate, ax_voltage = plot_tools.plot_network_activity(
        rate_monitor_inhib, spike_monitor_inhib, voltage_monitor_inhib, t_min=0. * b2.ms)
    fig.canvas.set_window_title('Inhib Population')
    plt.show()

    # return rate_monitor_A, spike_monitor_A, voltage_monitor_A, idx_monitored_neurons_A,\
    #        rate_monitor_B, spike_monitor_B, voltage_monitor_B, idx_monitored_neurons_B,\
    #        rate_monitor_inhib, spike_monitor_inhib, voltage_monitor_inhib, idx_monitored_neurons_inhib


def getting_started():
    # b2.defaultclock.dt = 0.1 * b2.ms
    b2.defaultclock.dt = 0.05 * b2.ms
    # rate_monitor_A, spike_monitor_A, voltage_monitor_A, idx_monitored_neurons_A, \
    #        rate_monitor_B, spike_monitor_B, voltage_monitor_B, idx_monitored_neurons_B,\
    #        rate_monitor_inhib, spike_monitor_inhib, voltage_monitor_inhib, idx_monitored_neurons_inhib = \
    simulate_wm(
        N_extern=1500, firing_rate_extern=5.8 * b2.Hz,
        N_Excit=800, w_pos=1.6,
        N_Inhib=200,
        sim_time=250. * b2.ms)

if __name__ == "__main__":
    getting_started()
