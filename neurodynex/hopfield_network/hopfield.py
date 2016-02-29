"""
This file implements a Hopfield Network model.

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

import matplotlib.pyplot as plt
import numpy as np
from copy import copy
import pickle
import gzip
from pkg_resources import resource_filename
import sys

plot_dic = {'cmap': plt.cm.gray, 'interpolation': 'nearest'}


class HopfieldNetwork:
    """Implements a Hopfield network of size N.

    Attributes:
        N (int): Square root of number of neurons
        patterns (numpy.ndarray): Array of stored patterns
        weight (numpy.ndarray): Array of stored weights
        x (numpy.ndarray): Network state (of size N**2)
    """

    def __init__(self, N):
        self.N = N

    def make_pattern(self, P=1, ratio=0.5, letters=None):
        """Creates and stores additional patterns to the
        network.

        Args:
            P (int, optional): number of patterns
                (used only for random patterns)
            ratio (float, optional): percentage of 'on' pixels
                for random patterns
            letters (TYPE, optional): to store characters
                use as input a string with the desired letters.
                Example: ``make_pattern(letters='abcdjft')``

        Raises:
            ValueError: Raised if N!=10 and letters!=None. For now
                letters are hardcoded for N=10.
        """

        if letters:
            if self.N != 10:
                raise ValueError('the network size must be equal to 10')
            alph = load_alphabet()
            self.patterns = -np.ones((len(letters), self.N**2), int)
            idx = 0
            for i in letters:
                self.patterns[idx] = alph[i]
                idx += 1
        else:
            self.patterns = -np.ones((P, self.N**2), int)
            idx = int(ratio*self.N**2)
            for i in range(P):
                self.patterns[i, :idx] = 1
                self.patterns[i] = np.random.permutation(self.patterns[i])

        self.weight = np.zeros((self.N**2, self.N**2))
        for i in range(self.N**2):
            self.weight[i] = 1./self.N**2 * (
                np.sum(
                    self.patterns[k, i] * self.patterns[k]
                    for k in range(self.patterns.shape[0])
                )
            )

    def grid(self, mu=None):
        """Reshape an array of length NxN to a matrix NxN

        Args:
            mu (TYPE, optional): If None, return the reshaped
                network state. For an integer i < P, return the
                reshaped pattern i.

        Returns:
            numpy.ndarray: Reshaped network state or pattern
        """

        if mu is not None:
            x_grid = np.reshape(self.patterns[mu], (self.N, self.N))
        else:
            x_grid = np.reshape(self.x, (self.N, self.N))
        return x_grid

    def dynamic(self):
        """Executes one timestep of the dynamics"""

        h = np.sum(self.weight*self.x, axis=1)
        self.x = np.sign(h)

    def overlap(self, mu):
        """Computes the overlap of the current state with
        pattern number mu.

        Args:
            mu (int): The index of the pattern to
                compare with.
        """

        return 1./self.N**2*np.sum(self.patterns[mu]*self.x)

    def run(self, t_max=20, mu=0, flip_ratio=0, do_plot=True):
        """Runs the dynamics and optionally plots it.

        Args:
            t_max (float, optional): Timesteps to simulate
            mu (int, optional): Pattern number to use
                as initial pattern for the network state (< P)
            flip_ratio (int, optional): ratio of randomized pixels.
                For example, to run pattern #5 with 5% flipped pixels use
                ``run(mu=5,flip_ratio=0.05)``
            do_plot (bool, optional): Plot the network as it is
                updated

        Raises:
            IndexError: Raised if given pattern index is too high.
            RuntimeError: Raised if no patterns have been created.
        """
        try:
            self.patterns
        except AttributeError:
            raise RuntimeError(
                'No patterns created: please ' +
                'use make_pattern to create at least one pattern.'
            )

        try:
            self.patterns[mu]
        except:
            raise IndexError('Pattern index too high (has to be < P)')

        # set the initial state of the net
        self.x = copy(self.patterns[mu])
        flip = np.random.permutation(np.arange(self.N**2))
        idx = int(self.N**2 * flip_ratio)
        self.x[flip[0:idx]] *= -1
        t = [0]
        overlap = [self.overlap(mu)]

        # prepare the figure
        fig = plt.figure()

        # plot the current network state
        plt.subplot(221)
        # keep a handle to the image for updating
        g1 = plt.imshow(self.grid(), **plot_dic)
        plt.axis('off')
        plt.title('x')

        # plot the target pattern
        plt.subplot(222)
        plt.imshow(self.grid(mu=mu), **plot_dic)
        plt.axis('off')
        plt.title('pattern %i' % mu)

        # plot the time course of the overlap
        plt.subplot(212)
        # keep a handle to the image for updating
        g2, = plt.plot(t, overlap, 'k', lw=2)
        plt.axis([0, t_max, -1, 1])
        plt.ylim([-1.1, 1.1])
        plt.xlabel('time step')
        plt.ylabel('overlap')

        # this forces pylab to update and show the fig.
        fig.show()
        x_old = copy(self.x)

        for i in range(t_max):

            # run a step
            self.dynamic()
            t.append(i+1)
            overlap.append(self.overlap(mu))

            # update the plotted data
            g1.set_data(self.grid())
            g2.set_data(t, overlap)

            # update the figure so that we see the changes
            plt.draw()

            # check the exit condition
            i_fin = i+1
            if np.sum(np.abs(x_old-self.x)) == 0:
                break
            x_old = copy(self.x)

            # sleep for replotting
            plt.pause(0.5)

        print("Pattern recovered in %i time steps." % i_fin +
              " Final overlap %.3f" % overlap[-1])


def load_alphabet():
    """Load alphabet dict from the file
    ``data/alphabet.pickle.gz``, which is included in
    the neurodynex release.

    Returns:
        dict: Dictionary of 10x10 patterns

    Raises:
        ImportError: Raised if ``neurodynex``
            can not be imported. Please install
            `neurodynex <pypi.python.org/pypi/neurodynex/>`_.
    """

    file_str = 'data/alphabet.pickle.gz'

    try:
        file_name = resource_filename('neurodynex', file_str)
    except ImportError:
        raise ImportError(
            "Could not import data file %s. " % file_str +
            "Make sure the pypi package `neurodynex` is installed!"
        )

    with gzip.open("%s" % file_name) as f:
        if sys.version_info < (3, 0, 0):
            # python2 pickle.loads has no attribute 'encoding'
            return pickle.load(f)
        else:
            # latin1 is required for python3 compatibility
            return pickle.load(f, encoding='latin1')
