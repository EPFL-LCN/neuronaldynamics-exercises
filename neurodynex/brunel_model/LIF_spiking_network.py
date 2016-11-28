"""
Implementation of the Brunel 2000 network:
sparsely connected network of identical LIF neurons (Model A).
"""
import brian2 as b2
from brian2 import NeuronGroup
import matplotlib.pyplot as plt


# Default parameters of a single LIF neuron
V_REST = 0. * b2.mV
V_RESET = +10. * b2.mV
FIRING_THRESHOLD = +20. * b2.mV
MEMBRANE_TIME_SCALE = 20. * b2.ms
ABSOLUTE_REFRACTORY_PERIOD = 2.0 * b2.ms

# Default parametes of the network
SYNAPTIC_WEIGHT_W0 = 0.1 * b2.mV
# note: w_ee = w_ei = w0 and w_ie=w_ii = -g*w0
RELATIVE_INHIBITORY_STRENGTH_G = 3

CONNECTION_PROBABILITY_EPSILON = 0.1

def Brunel_Network(
        N_E=1000,
        N_I=250,
        connection_probability=CONNECTION_PROBABILITY_EPSILON,
        w0 = SYNAPTIC_WEIGHT_W0,
        g = RELATIVE_INHIBITORY_STRENGTH_G,

        v_rest=V_REST,
        v_reset=V_RESET,
        firing_threshold=FIRING_THRESHOLD,
        membrane_time_scale=MEMBRANE_TIME_SCALE,
        abs_refractory_period=ABSOLUTE_REFRACTORY_PERIOD):
    print("start")

    lif_dynamics = """
    dv/dt = -(v-v_rest) / membrane_time_scale : volt (unless refractory)"""


    all_neurons = NeuronGroup(N_E+N_I, model=lif_dynamics, threshold='v > vt',
                          reset='v=el', refractory=5*b2.ms, method='euler')
    excitatory_population = all_neurons[:N_E]
    inhibitory_population = all_neurons[N_E:]

if __name__ == "__main__":
    Brunel_Network()

