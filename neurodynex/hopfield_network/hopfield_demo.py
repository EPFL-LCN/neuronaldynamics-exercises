import hf_plot_tools as hfplot
import hf_pattern_tools as pattern_tools
import hf_network
import matplotlib.pyplot as plt
import numpy as np


def run_hf_demo(pattern_size=4, nr_random_patterns=3, reference_pattern=0,
                initially_flipped_pixels=3, nr_iterations=6, random_seed=None):
    # instantiate a hofpfield network
    hopfield_net = hf_network.HopfieldNetwork(pattern_size**2)
    # instantiate a pattern factory
    factory = pattern_tools.PatternFactory(pattern_size, pattern_size)
    # create a checkerboard pattern and add it to the pattern list
    checkerboard = factory.create_checkerboard()
    pattern_list = [checkerboard]
    # add random patterns to the list
    pattern_list.extend(factory.create_random_pattern_list(nr_random_patterns, on_probability=0.5))
    hfplot.plot_pattern_list(pattern_list)
    # let the hopfield network 'learn' the patterns. Note: they are not stored
    # explicitly but only network weights are updated !
    hopfield_net.store_patterns(pattern_list)
    # uncomment the following line to enable a probabilistic network dynamic
    # hopfield_net.set_probabilistic_update(2.5)

    # how similar are the random patterns? Check the overlaps
    overlap_matrix = pattern_tools.compute_overlap_matrix(pattern_list)
    hfplot.plot_overlap_matrix(overlap_matrix)
    # create a noisy version of a pattern and use that to initialize the network
    noisy_init_state = pattern_tools.flip_n(pattern_list[reference_pattern], initially_flipped_pixels)
    hopfield_net.set_state_from_pattern(noisy_init_state)
    # run the network dynamics and record the network state at every time step
    states = hopfield_net.run_with_monitoring(nr_iterations)
    # each network state is a vector. reshape it to the same shape used to create the patterns.
    states_as_patterns = factory.reshape_patterns(states)
    # plot the states of the network
    hfplot.plot_state_sequence_and_overlap(states_as_patterns, pattern_list, reference_pattern)
    plt.show()


def run_hf_demo_alphabet(letters, initialization_noise_level=0.2, random_seed=None):
    import numpy as np
    # in your code, don't forget to import hf_plot_tools

    # fixed size 10 for the alphabet.
    pattern_size = 10
    # pick some letters we want to store in the network
    if letters is None:
        letters = ['a', 'b', 'c', 'r', 's', 'x', 'y', 'z']
    reference_pattern = 0

    # instantiate a hofpfield network
    hopfield_net = hf_network.HopfieldNetwork(pattern_size**2)
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

run_hf_demo()

# Demo2: more neurons, more patterns, more noise
run_hf_demo(pattern_size=6, nr_random_patterns=5, initially_flipped_pixels=11, nr_iterations=5)

# Demo3: more parameters
# run_hf_demo(pattern_size=4, nr_random_patterns=5,
#                 reference_pattern=0, initially_flipped_pixels=4, nr_iterations=6,
#                 random_seed=50)

print('recover letter A')
letter_list = ['a', 'b', 'c', 's', 'x', 'y', 'z']
run_hf_demo_alphabet(letter_list, initialization_noise_level=0.2, random_seed=76)

print('letter A not recovered despite the overlap m = 1 after one iteration')
letter_list.append('r')
run_hf_demo_alphabet(letter_list, initialization_noise_level=0.2, random_seed=76)
