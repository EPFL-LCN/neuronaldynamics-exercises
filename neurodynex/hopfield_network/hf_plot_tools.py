"""
Helper tools to visualize patterns and network state
"""
import matplotlib.pyplot as plt
import hf_pattern_tools as pattern_tools
import numpy as np


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


def plot_overlap_matrix(overlap_matrix):
    plt.imshow(overlap_matrix, interpolation='nearest', cmap='bwr')
    plt.title("pattern overlap m(i,k)")
    plt.xlabel("pattern k")
    plt.ylabel("pattern i")
    plt.axes().get_xaxis().set_major_locator(plt.MaxNLocator(integer=True))
    plt.axes().get_yaxis().set_major_locator(plt.MaxNLocator(integer=True))
    cb = plt.colorbar(ticks=np.arange(-1, 1.01, 0.25).tolist())
    cb.set_clim(-1, 1)
    plt.show()


def plot_state_sequence(state_sequence, reference=None):
    """
    Args:
        state_sequence:
        reference:
    """
    f, ax = plt.subplots(1, len(state_sequence))
    _plot_list(ax, state_sequence, None, "P{0}")
    plt.show()


def plot_pattern_list(pattern_list):
    f, ax = plt.subplots(1, len(pattern_list))
    _plot_list(ax, pattern_list, None, "P{0}")
    plt.show()


def _plot_list(axes_list, state_sequence, reference=None, title_pattern="S({0})"):
    """
    For internal use.
    Plots all states S(t) or patterns P in state_sequence.
    If a (optional) reference pattern is provided, the patters are  plotted with differences highlighted
    Args:
        state_sequence: (list(numpy.ndarray))
        reference: (numpy.ndarray)
        title_pattern (str) pattern injecting index i
    """
    for i in range(len(state_sequence)):
        if reference is None:
            p = state_sequence[i]
        else:
            p = pattern_tools.get_pattern_diff(state_sequence[i], reference, diff_code=-0.2)
        axes_list[i].imshow(p, interpolation='nearest', cmap='hot')
        axes_list[i].set_title(title_pattern.format(i))
        axes_list[i].axis('off')


def plot_state_sequence_and_overlap(state_sequence, pattern_list, reference_idx):
    """
    For each time point t (=index of state_sequence), plots the sequence of states and the overlap (barplot)
    between state(t) and each pattern.
    Args:
        state_sequence: (list(numpy.ndarray))
        pattern_list: (list(numpy.ndarray))
        reference_idx: (int) identifies the pattern in pattern_list for which wrong pixels are colored.
    """
    if reference_idx is None:
        reference_idx = 0
    reference = pattern_list[reference_idx]
    f, ax = plt.subplots(2, len(state_sequence))
    _plot_list(ax[0, :], state_sequence, reference, "S{0}")
    for i in range(len(state_sequence)):
        overlap_list = pattern_tools.compute_overlap_list(state_sequence[i], pattern_list)
        ax[1, i].bar(range(len(overlap_list)), overlap_list)
        ax[1, i].set_title("m={1}".format(i, round(overlap_list[reference_idx], 2)))
        ax[1, i].set_ylim([-1, 1])
        ax[1, i].get_xaxis().set_major_locator(plt.MaxNLocator(integer=True))
        if i > 0:  # show lables only for the first subplot
            ax[1, i].set_xticklabels([])
            ax[1, i].set_yticklabels([])
    plt.show()
