"""
Helper tools to visualize patterns and network state
"""
import matplotlib.pyplot as plt
import hf_pattern_tools as pattern_tools
import hf_network


def plot_pattern(pattern, reference=None):
    """
    Plots the pattern. If a (optional) reference pattern is provided, the pattern is  plotted
     with differences highlighted
    Args:
        pattern (numpy.ndarray): N by N pattern to plot
        reference (numpy.ndarray):  optional. If set, differences between pattern and reference are highlighted
    """
    plt.figure()
    if reference is None:
        p = pattern
        overlap = 1
    else:
        p = pattern_tools.get_pattern_diff(pattern, reference)
        overlap = pattern_tools.compute_overlap(pattern, reference)

    plt.imshow(p, interpolation='nearest', cmap='hot')
    if reference is not None:
        plt.title("m={:0.2f}".format(round(overlap, 2)))
    plt.axis('off')
    plt.show()


def plot_state_sequence(state_sequence, reference=None):
    n_patterns = len(state_sequence)
    plt.figure()
    for i in range(n_patterns):
        plt.subplot(1, n_patterns, i + 1)
        if reference is None:
            p = state_sequence[i]
        else:
            p = pattern_tools.get_pattern_diff(state_sequence[i], reference)
            overlap = pattern_tools.compute_overlap(state_sequence[i], reference)
            plt.title("m={:0.2f}".format(round(overlap, 2)))
        plt.imshow(p, interpolation='nearest', cmap='hot')
        plt.axis('off')
    plt.show()


def plot_state_sequence_and_overlap(state_sequence, pattern_list, reference_idx):
    if reference_idx is None:
        reference_idx = 0
    reference = pattern_list[reference_idx]
    n_states = len(state_sequence)
    plt.figure()
    for i in range(n_states):
        pattern_with_diffs = pattern_tools.get_pattern_diff(state_sequence[i], reference, -0.2)
        overlap = pattern_tools.compute_overlap(state_sequence[i], reference)

        plt.subplot(2, n_states, i + 1)
        plt.imshow(pattern_with_diffs, interpolation='nearest', cmap='hot')
        plt.title("m={:0.2f}".format(round(overlap, 2)))
        plt.axis('off')

        plt.subplot(2, n_states, n_states + i + 1)
        overlap_list = pattern_tools.compute_overlap_list(state_sequence[i], pattern_list)
        plt.bar(range(len(overlap_list)), overlap_list)
        plt.ylim([-1, 1])
    plt.show()


def plot_demo(initially_flipped_pixels=4, nr_random_patterns=4):
    import numpy as np
    # in your code, don't forget to import hf_plot_tools

    pattern_size = 4
    # we initialize the network with pattern nr1, but 2 pixels flipped
    reference_pattern = 0

    # instantiate a hofpfield network
    hopfield_net = hf_network.HopfieldNetwork(pattern_size)
    # instantiate a pattern factory
    factory = pattern_tools.PatternFactory(pattern_size)
    # for the demo, use a seed to get a reproducible pattern
    # np.random.seed(39)
    np.random.seed(52)
    checkerboard = factory.create_checkerboard()
    plot_pattern(checkerboard)
    pattern_list = [checkerboard]
    pattern_list.extend(factory.create_random_pattern_list(nr_random_patterns, on_probability=0.5))
    hopfield_net.store_patterns(pattern_list)
    plot_state_sequence(pattern_list)
    overlap_matrix = pattern_tools.compute_overlap_matrix(pattern_list)
    # how similar are the random patterns? Check the overlaps
    print(overlap_matrix)
    noisy_init_state = pattern_tools.flip_n(pattern_list[reference_pattern], initially_flipped_pixels)
    hopfield_net.set_state_from_2d_pattern(noisy_init_state)
    states = hopfield_net.run_with_monitoring(5)
    plot_state_sequence_and_overlap(states, pattern_list, reference_pattern)


def plot_demo_2(pattern_size=4, nr_patterns=6, reference_pattern=0, initially_flipped_pixels=3, nr_iterations=6):
    import numpy as np
    # in your code, don't forget to import hf_plot_tools

    # instantiate a hofpfield network
    hopfield_net = hf_network.HopfieldNetwork(pattern_size)
    # instantiate a pattern factory
    factory = pattern_tools.PatternFactory(pattern_size)
    # for the demo, use a seed to get a reproducible pattern
    np.random.seed(39)
    pattern_list = factory.create_random_pattern_list(nr_patterns, on_probability=0.5)
    plot_state_sequence(pattern_list)

    hopfield_net.store_patterns(pattern_list)
    overlap_matrix = pattern_tools.compute_overlap_matrix(pattern_list)
    # how similar are the random patterns? Check the overlaps
    print(overlap_matrix)
    plt.imshow(overlap_matrix, interpolation='nearest', cmap='bwr')
    cb = plt.colorbar()
    # cb.set_clim(-1, +1)
    plt.show()
    noisy_init_state = pattern_tools.flip_n(pattern_list[reference_pattern], initially_flipped_pixels)
    hopfield_net.set_state_from_2d_pattern(noisy_init_state)
    states = hopfield_net.run_with_monitoring(nr_iterations)
    plot_state_sequence_and_overlap(states, pattern_list, reference_pattern)


def plot_demo_alphabet(letters, initialization_noise_level=0.2):
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
    np.random.seed(76)
    # load the dictionary
    abc_dict = pattern_tools.load_alphabet()
    # for each key in letters, append the pattern to the list
    pattern_list = [abc_dict[key] for key in letters]
    plot_state_sequence(pattern_list)

    hopfield_net.store_patterns(pattern_list)
    hopfield_net.set_state_from_2d_pattern(
        pattern_tools.get_noisy_copy(abc_dict[letters[reference_pattern]], initialization_noise_level))
    states = hopfield_net.run_with_monitoring(6)
    plot_state_sequence_and_overlap(states, pattern_list, reference_pattern)
