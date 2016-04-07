"""
This file implements a Hopfield Network network. It provides functions to
set and retrieve the network state, store patterns.

Relevant book chapters:
    - http://neuronaldynamics.epfl.ch/online/Ch17.S2.html

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

import numpy as np
import math


class HopfieldNetwork:
    """Implements a Hopfield network.

    Attributes:
        nrOfNeurons (int): Number of neurons
        weights (numpy.ndarray): nrOfNeurons x nrOfNeurons matrix of weights
        state (numpy.ndarray): current network state (of size N**2)
    """

    def __init__(self, nr_neurons):
        """
        Constructor
        Args:
            nr_neurons (int): Number of neurons. Use a square number to get the
            visualizations properly
        """
        math.sqrt(nr_neurons)
        self.nrOfNeurons = nr_neurons
        self._grid_width = int(math.floor(math.sqrt(nr_neurons)))
        self.state = 2 * np.random.randint(0, 2, self.nrOfNeurons) - 1
        # initialize random weights
        self.weights = 0
        self.reset_weights()
        self._update_method = _get_sign_update_function()

    def reset_weights(self):
        """
        Resets the weights to random values
        """
        self.weights = 1.0 / self.nrOfNeurons * \
                       (2 * np.random.rand(self.nrOfNeurons, self.nrOfNeurons) - 1)

    def set_probabilistic_update(self, inverse_temp_beta):
        self._update_method = _get_probabilistic_update_function(inverse_temp_beta)

    def set_update_method(self, update_method):
        self._update_method = update_method

    def store_patterns(self, pattern_list):
        """
        Learns the patterns by setting the network weights. The patterns
        themselves are not stored, only the weights are updated!
        self connections are set to 0.
        Args:
            pattern_list: a nonempty list of patterns.
            Make sure sure self.nrOfNeurons = len(pattern)
        """
        self.weights = np.zeros((self.nrOfNeurons, self.nrOfNeurons))
        # textbook formula to compute the weights:
        for p in pattern_list:
            p_flat = p.flatten()
            for i in range(self.nrOfNeurons):
                for k in range(self.nrOfNeurons):
                    self.weights[i, k] += p_flat[i] * p_flat[k]
        self.weights /= self.nrOfNeurons
        # no self connections:
        np.fill_diagonal(self.weights, 0)

    def get_state2d(self):
        """
        The state is a vector of length N*N. It is more convenient to use a
        2-D representation of the state.
        Returns:
            an N by N ndarray of the network state.
        """
        return self.state.copy().reshape((self._grid_width, self._grid_width))

    def set_state_from_2d_pattern(self, pattern):
        self.state = pattern.copy().reshape(self.nrOfNeurons)

    def iterate(self):
        """Executes one timestep of the dynamics"""
        self.state = self._update_method(self.state, self.weights)

    def run(self, t_max=5):
        """Runs the dynamics.
        Args:
            t_max (float, optional): Timesteps to simulate
        """
        for i in range(t_max):
            # run a step
            self.iterate()

    def run_with_monitoring(self, t_max=5):
        """
        Iterates at most t_max steps. records the network state after every
        iteration
        Args:
            t_max:

        Returns:
            a list of 2d network states
        """
        states = list()
        states.append(self.get_state2d())
        for i in range(t_max):
            # run a step
            self.iterate()
            states.append(self.get_state2d())
        return states


def _get_sign_update_function():
    """
    for internal use
    Returns:
        A function implementing a synchronous state update using sigmoid
    """

    def upd(state_s0, weights):
        h = np.sum(weights * state_s0, axis=1)
        s1 = np.sign(h)
        # by definition, neurons have state +/-1. If the
        # sign function returns 0, we set it to +1
        idx0 = s1 == 0
        s1[idx0] = 1
        return s1

    return upd


def _get_probabilistic_update_function(inverse_temp_beta):
    """
    for internal use
    Args:
        inverse_temp_beta (float)>=0:

    Returns:
        A function implementing a probabilistic, synchronous state update
        using a sigmoidal transfer function g:= tanh(beta*h).
    """

    def upd(state_s0, weights):
        h = np.sum(weights * state_s0, axis=1)
        g = np.tanh(inverse_temp_beta * h)  # closure: inverse_temp_beta is available
        prob = 0.5 * (g + 1.0)
        s = []
        for p in prob:
            s.append(2 * np.random.binomial(1, p) - 1)
        s1 = np.asarray(s)
        return s1

    return upd
