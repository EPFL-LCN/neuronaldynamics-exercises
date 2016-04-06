import hf_plot_tools as hfplot
import hf_pattern_tools as pattern_tools
import hf_network
import matplotlib.pyplot as plt


def plot_demo(pattern_size=4, nr_random_patterns=3, reference_pattern=0,
              initially_flipped_pixels=3, nr_iterations=6, random_seed=None):
    # instantiate a hofpfield network
    hopfield_net = hf_network.HopfieldNetwork(pattern_size)
    # instantiate a pattern factory
    factory = pattern_tools.PatternFactory(pattern_size)
    checkerboard = factory.create_checkerboard()
    pattern_list = [checkerboard]
    pattern_list.extend(factory.create_random_pattern_list(nr_random_patterns, on_probability=0.5))
    hfplot.plot_pattern_list(pattern_list)
    hopfield_net.store_patterns(pattern_list)
    overlap_matrix = pattern_tools.compute_overlap_matrix(pattern_list)
    # how similar are the random patterns? Check the overlaps
    hfplot.plot_overlap_matrix(overlap_matrix)
    noisy_init_state = pattern_tools.flip_n(pattern_list[reference_pattern], initially_flipped_pixels)
    hopfield_net.set_state_from_2d_pattern(noisy_init_state)
    states = hopfield_net.run_with_monitoring(nr_iterations)
    hfplot.plot_state_sequence_and_overlap(states, pattern_list, reference_pattern)
    plt.show()


def plot_demo_alphabet(letters, initialization_noise_level=0.2, random_seed=None):
    import numpy as np
    # in your code, don't forget to import hf_plot_tools

    # fixed size 10 for the alphabet.
    pattern_size = 10
    # pick some letters we want to store in the network
    if letters is None:
        letters = ['a', 'b', 'c', 'r', 's', 'x', 'y', 'z']
    reference_pattern = 0

    # instantiate a hofpfield network
    hopfield_net = hf_network.HopfieldNetwork(pattern_size)
    # for the demo, use a seed to get a reproducible pattern
    np.random.seed(random_seed)
    # load the dictionary
    abc_dict = pattern_tools.load_alphabet()
    # for each key in letters, append the pattern to the list
    pattern_list = [abc_dict[key] for key in letters]
    hfplot.plot_pattern_list(pattern_list)

    hopfield_net.store_patterns(pattern_list)
    hopfield_net.set_state_from_2d_pattern(
        pattern_tools.get_noisy_copy(abc_dict[letters[reference_pattern]], initialization_noise_level))
    states = hopfield_net.run_with_monitoring(6)
    hfplot.plot_state_sequence_and_overlap(states, pattern_list, reference_pattern)

plot_demo()

# Demo2: more neurons, more patterns, more noise
plot_demo(pattern_size=6, nr_random_patterns=5, initially_flipped_pixels=9)

# Demo3: more parameters
# plot_demo(pattern_size=4, nr_random_patterns=5,
#                 reference_pattern=0, initially_flipped_pixels=4, nr_iterations=6,
#                 random_seed=50)

print('recover letter A')
letter_list = ['a', 'b', 'c', 's', 'x', 'y', 'z']
plot_demo_alphabet(letter_list, initialization_noise_level=0.2, random_seed=76)

print('letter A not recovered despite the overlap m = 1 after one iteration')
letter_list.append('r')
plot_demo_alphabet(letter_list, initialization_noise_level=0.2, random_seed=76)
