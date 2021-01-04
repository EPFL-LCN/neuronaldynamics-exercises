"""
Helper tools to visualize patterns and network state
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

import matplotlib.pyplot as plt
import neurodynex3.hopfield_network.pattern_tools as pattern_tools
import numpy as np


def plot_pattern(pattern, reference=None, color_map="brg", diff_code=0):
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
        p = pattern_tools.get_pattern_diff(pattern, reference, diff_code)
        overlap = pattern_tools.compute_overlap(pattern, reference)

    plt.imshow(p, interpolation="nearest", cmap=color_map)
    if reference is not None:
        plt.title("m = {:0.2f}".format(round(overlap, 2)))
    plt.axis("off")
    plt.show()


def plot_overlap_matrix(overlap_matrix, color_map="bwr"):
    """
    Visualizes the pattern overlap

    Args:
        overlap_matrix:
        color_map:

    """

    plt.imshow(overlap_matrix, interpolation="nearest", cmap=color_map)
    plt.title("pattern overlap m(i,k)")
    plt.xlabel("pattern k")
    plt.ylabel("pattern i")
    plt.axes().get_xaxis().set_major_locator(plt.MaxNLocator(integer=True))
    plt.axes().get_yaxis().set_major_locator(plt.MaxNLocator(integer=True))
    cb = plt.colorbar(ticks=np.arange(-1, 1.01, 0.25).tolist())
    cb.set_clim(-1, 1)
    plt.show()


def plot_pattern_list(pattern_list, color_map="brg"):
    """
    Plots the list of patterns

    Args:
        pattern_list:
        color_map:

    Returns:

    """
    f, ax = plt.subplots(1, len(pattern_list))
    if len(pattern_list) == 1:
        ax = [ax]  # for n=1, subplots() does not return a list
    _plot_list(ax, pattern_list, None, "P{0}", color_map)
    plt.show()


def _plot_list(axes_list, state_sequence, reference=None, title_pattern="S({0})", color_map="brg"):
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
        if np.max(p) == np.min(p):
            axes_list[i].imshow(p, interpolation="nearest", cmap='RdYlBu')
        else:
            axes_list[i].imshow(p, interpolation="nearest", cmap=color_map)
        axes_list[i].set_title(title_pattern.format(i))
        axes_list[i].axis("off")


def plot_state_sequence_and_overlap(state_sequence, pattern_list, reference_idx, color_map="brg", suptitle=None):
    """
    For each time point t ( = index of state_sequence), plots the sequence of states and the overlap (barplot)
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
    if len(state_sequence) == 1:
        ax = [ax]
    print()
    _plot_list(ax[0, :], state_sequence, reference, "S{0}", color_map)
    for i in range(len(state_sequence)):
        overlap_list = pattern_tools.compute_overlap_list(state_sequence[i], pattern_list)
        ax[1, i].bar(range(len(overlap_list)), overlap_list)
        ax[1, i].set_title("m = {1}".format(i, round(overlap_list[reference_idx], 2)))
        ax[1, i].set_ylim([-1, 1])
        ax[1, i].get_xaxis().set_major_locator(plt.MaxNLocator(integer=True))
        if i > 0:  # show lables only for the first subplot
            ax[1, i].set_xticklabels([])
            ax[1, i].set_yticklabels([])
    if suptitle is not None:
        f.suptitle(suptitle)
    plt.show()


def plot_network_weights(hopfield_network, color_map="jet"):
    """
    Visualizes the network's weight matrix

    Args:
        hopfield_network:
        color_map:

    """

    plt.figure()
    plt.imshow(hopfield_network.weights, interpolation="nearest", cmap=color_map)
    plt.colorbar()
