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
        math.sqrt(nr_neurons)
        self.nrOfNeurons = nr_neurons
        self._grid_width = int(math.floor(math.sqrt(nr_neurons)))
        self.state = 2*np.random.randint(0, 2, self.nrOfNeurons)-1
        # initialize random weights
        self.weights = 0
        self.reset_weights()

    def reset_weights(self):
        self.weights = 1.0/self.nrOfNeurons * \
                       (2*np.random.rand(self.nrOfNeurons, self.nrOfNeurons)-1)

    def store_patterns(self, pattern_list):
        """
        Learns the patterns by setting the network weights. The patterns
        themselves are not stored!
        Args:
            pattern_list: a nonempty list of 2d patterns.
        """
        self.weights = np.zeros((self.nrOfNeurons, self.nrOfNeurons))
        # textbook formula to compute the weights:
        for p in pattern_list:
            p_flat = p.flatten()
            for i in range(self.nrOfNeurons):
                for k in range(self.nrOfNeurons):
                    self.weights[i, k] += p_flat[i]*p_flat[k]
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
        h = np.sum(self.weights*self.state, axis=1)
        self.state = np.sign(h)
        # by definition, neurons have state +/-1. If the
        # sign function returns 0, we set 0 to +1
        idx0 = self.state == 0
        self.state[idx0] = 1

        # # probabilistic update:
        # g = np.tanh(2.6*h)
        # prob = 0.5*(g+1)
        # s = []
        # for p in prob:
        #     s.append(2*np.random.binomial(1, p)-1)
        # self.state = np.asarray(s)

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
