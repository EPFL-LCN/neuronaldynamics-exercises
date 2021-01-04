
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

import neurodynex3.hopfield_network.plot_tools as hfplot
import neurodynex3.hopfield_network.pattern_tools as pattern_tools
import neurodynex3.hopfield_network.network as network
import matplotlib.pyplot as plt
import numpy as np


def run_hf_demo(pattern_size=4, nr_random_patterns=3, reference_pattern=0,
                initially_flipped_pixels=3, nr_iterations=6, random_seed=None):
    """
    Simple demo.

    Args:
        pattern_size:
        nr_random_patterns:
        reference_pattern:
        initially_flipped_pixels:
        nr_iterations:
        random_seed:

    Returns:

    """
    # instantiate a hofpfield network
    hopfield_net = network.HopfieldNetwork(pattern_size**2)

    # for the demo, use a seed to get a reproducible pattern
    np.random.seed(random_seed)

    # instantiate a pattern factory
    factory = pattern_tools.PatternFactory(pattern_size, pattern_size)
    # create a checkerboard pattern and add it to the pattern list
    checkerboard = factory.create_checkerboard()
    pattern_list = [checkerboard]
    # add random patterns to the list
    pattern_list.extend(factory.create_random_pattern_list(nr_random_patterns, on_probability=0.5))
    hfplot.plot_pattern_list(pattern_list)
    # let the hopfield network "learn" the patterns. Note: they are not stored
    # explicitly but only network weights are updated !
    hopfield_net.store_patterns(pattern_list)

    # how similar are the random patterns? Check the overlaps
    overlap_matrix = pattern_tools.compute_overlap_matrix(pattern_list)
    hfplot.plot_overlap_matrix(overlap_matrix)
    # create a noisy version of a pattern and use that to initialize the network
    noisy_init_state = pattern_tools.flip_n(pattern_list[reference_pattern], initially_flipped_pixels)
    hopfield_net.set_state_from_pattern(noisy_init_state)

    # uncomment the following line to enable a PROBABILISTIC network dynamic
    # hopfield_net.set_dynamics_probabilistic_sync(2.5)
    # uncomment the following line to enable an ASYNCHRONOUS network dynamic
    # hopfield_net.set_dynamics_sign_async()

    # run the network dynamics and record the network state at every time step
    states = hopfield_net.run_with_monitoring(nr_iterations)
    # each network state is a vector. reshape it to the same shape used to create the patterns.
    states_as_patterns = factory.reshape_patterns(states)
    # plot the states of the network
    hfplot.plot_state_sequence_and_overlap(states_as_patterns, pattern_list, reference_pattern)
    plt.show()


def run_hf_demo_alphabet(letters, initialization_noise_level=0.2, random_seed=None):
    """
    Simple demo

    Args:
        letters:
        initialization_noise_level:
        random_seed:

    Returns:

    """

    # fixed size 10 for the alphabet.
    pattern_size = 10
    # pick some letters we want to store in the network
    if letters is None:
        letters = ['A', 'B', 'C', 'R', 'S', 'X', 'Y', 'Z']
    reference_pattern = 0

    # instantiate a hofpfield network
    hopfield_net = network.HopfieldNetwork(pattern_size**2)
    # for the demo, use a seed to get a reproducible pattern
    np.random.seed(random_seed)
    # load the dictionary
    abc_dict = pattern_tools.load_alphabet()
    # for each key in letters, append the pattern to the list
    pattern_list = [abc_dict[key] for key in letters]
    hfplot.plot_pattern_list(pattern_list)

    hopfield_net.store_patterns(pattern_list)
    hopfield_net.set_state_from_pattern(
        pattern_tools.get_noisy_copy(abc_dict[letters[reference_pattern]], initialization_noise_level))
    states = hopfield_net.run_with_monitoring(6)
    state_patterns = pattern_tools.reshape_patterns(states, pattern_list[0].shape)
    hfplot.plot_state_sequence_and_overlap(state_patterns, pattern_list, reference_pattern)
    plt.show()


def run_demo():
    """
    Simple demo

    """
    # Demo2: more neurons, more patterns, more noise
    # run_hf_demo(pattern_size=6, nr_random_patterns=5, initially_flipped_pixels=11, nr_iterations=5)

    # Demo3: more parameters
    # run_hf_demo(pattern_size=4, nr_random_patterns=5,
    #                 reference_pattern=0, initially_flipped_pixels=4, nr_iterations=6,
    #                 random_seed=50)

    print('recover letter A')
    letter_list = ['A', 'B', 'C', 'S', 'X', 'Y', 'Z']
    run_hf_demo_alphabet(letter_list, initialization_noise_level=0.2, random_seed=76)

    print('letter A not recovered despite the overlap m=1 after one iteration')
    letter_list.append('R')
    run_hf_demo_alphabet(letter_list, initialization_noise_level=0.2, random_seed=76)


def run_user_function_demo():
    def upd_random(state_s0, weights):
        nr_neurons = len(state_s0)
        random_neuron_idx_list = np.random.permutation(int(len(state_s0)/2))
        state_s1 = state_s0.copy()
        for i in range(len(random_neuron_idx_list)):
            state_s1[i] = -1 if (np.random.rand() < .5) else +1
        return state_s1

    hopfield_net = network.HopfieldNetwork(6**2)
    hopfield_net.set_dynamics_to_user_function(upd_random)

    # for the demo, use a seed to get a reproducible pattern
    # instantiate a pattern factory
    factory = pattern_tools.PatternFactory(6, 6)
    # create a checkerboard pattern and add it to the pattern list
    checkerboard = factory.create_checkerboard()
    pattern_list = [checkerboard]
    # add random patterns to the list
    pattern_list.extend(factory.create_random_pattern_list(4, on_probability=0.5))
    hfplot.plot_pattern_list(pattern_list)
    # let the hopfield network "learn" the patterns. Note: they are not stored
    # explicitly but only network weights are updated !
    hopfield_net.store_patterns(pattern_list)
    hopfield_net.set_state_from_pattern(pattern_list[0])

    # uncomment the following line to enable a PROBABILISTIC network dynamic
    # hopfield_net.set_dynamics_probabilistic_sync(2.5)
    # uncomment the following line to enable an ASYNCHRONOUS network dynamic
    # hopfield_net.set_dynamics_sign_async()

    # run the network dynamics and record the network state at every time step
    states = hopfield_net.run_with_monitoring(5)
    # each network state is a vector. reshape it to the same shape used to create the patterns.
    states_as_patterns = factory.reshape_patterns(states)
    # plot the states of the network
    hfplot.plot_state_sequence_and_overlap(states_as_patterns, pattern_list, 0)
    plt.show()


if __name__ == '__main__':
    run_demo()
    # run_user_function_demo()
