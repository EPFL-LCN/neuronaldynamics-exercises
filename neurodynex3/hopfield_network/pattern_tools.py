"""
Functions to create 2D patterns.
Note, in the hopfield model, we define patterns as vectors. To make
the exercise more visual, we use 2D patterns (N by N ndarrays).
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
from scipy import linalg
import pickle
import gzip
from pkg_resources import resource_filename
import sys


class PatternFactory:
    """
    Creates square patterns of size pattern_length x pattern_width
    If pattern length is omitted, square patterns are produced
    """
    def __init__(self, pattern_length, pattern_width=None):
        """
        Constructor
        Args:
            pattern_length: the length of a pattern
            pattern_width: width or None. If None, patterns are squares of size (pattern_length x pattern_length)
        """
        self.pattern_length = pattern_length
        self.pattern_width = pattern_length if pattern_width is None else pattern_width

    def create_random_pattern(self, on_probability=0.5):
        """
        Creates a pattern_length by pattern_width 2D random pattern
        Args:
            on_probability:

        Returns:
            a new random pattern
        """
        p = np.random.binomial(1, on_probability, self.pattern_length * self.pattern_width)
        p = p * 2 - 1  # map {0, 1} to {-1 +1}
        return p.reshape((self.pattern_length, self.pattern_width))

    def create_random_pattern_list(self, nr_patterns, on_probability=0.5):
        """
        Creates a list of nr_patterns random patterns
        Args:
            nr_patterns: length of the new list
            on_probability:

        Returns:
            a list of new random patterns of size (pattern_length x pattern_width)
        """
        p = list()
        for i in range(nr_patterns):
            p.append(self.create_random_pattern(on_probability))
        return p

    def create_row_patterns(self, nr_patterns=None):
        """
        creates a list of n patterns, the i-th pattern in the list
        has all states of the i-th row set to active.
        This is convenient to create a list of orthogonal patterns which
        are easy to visually identify

        Args:
            nr_patterns:

        Returns:
            list of orthogonal patterns
        """
        n = self.pattern_width if nr_patterns is None else nr_patterns
        pattern_list = []
        for i in range(n):
            p = self.create_all_off()
            p[i, :] = np.ones((1, self.pattern_length))
            pattern_list.append(p)
        return pattern_list

    def create_all_on(self):
        """
        Returns:
            2d pattern, all pixels on
        """
        return np.ones((self.pattern_length, self.pattern_width), np.int)

    def create_all_off(self):
        """
        Returns:
            2d pattern, all pixels off
        """
        return -1 * np.ones((self.pattern_length, self.pattern_width), np.int)

    def create_checkerboard(self):
        """
        creates a checkerboard pattern of size (pattern_length x pattern_width)
        Returns:
            checkerboard pattern
        """
        pw = np.ones(self.pattern_length, np.int)
        # set every second value to -1
        pw[1::2] = -1
        pl = np.ones(self.pattern_width, np.int)
        # set every second value to -1
        pl[1::2] = -1
        t = linalg.toeplitz(pw, pl)
        t = t.reshape((self.pattern_length, self.pattern_width))
        return t

    def create_L_pattern(self, l_width=1):
        """
        creates a pattern with column 0 (left) and row n (bottom) set to +1.
        Increase l_width to set more columns and rows (default is 1)

        Args:
            l_width (int): nr of rows and columns to set

        Returns:
            an L shaped pattern.
        """
        l_pat = -1 * np.ones((self.pattern_length, self.pattern_width), np.int)
        for i in range(l_width):
            l_pat[-i - 1, :] = np.ones(self.pattern_length, np.int)
            l_pat[:, i] = np.ones(self.pattern_length, np.int)
        return l_pat

    def reshape_patterns(self, pattern_list):
        """
        reshapes all patterns in pattern_list to have shape = (self.pattern_length, self.pattern_width)

        Args:
            self:
            pattern_list:

        Returns:

        """
        new_shape = (self.pattern_length, self.pattern_width)
        return reshape_patterns(pattern_list, new_shape)


def reshape_patterns(pattern_list, shape):
    """
    reshapes each pattern in pattern_list to the given shape

    Args:
        pattern_list:
        shape:

    Returns:

    """
    reshaped_patterns = [p.reshape(shape) for p in pattern_list]
    return reshaped_patterns


def get_pattern_diff(pattern1, pattern2, diff_code=0):
    """
    Creates a new pattern of same size as the two patterns.
    the diff pattern has the values pattern1 = pattern2 where the two patterns have
    the same value. Locations that differ between the two patterns are set to
    diff_code (default = 0)

    Args:
        pattern1:
        pattern2:
        diff_code: the values of the new pattern, at locations that differ between
        the two patterns are set to diff_code.
    Returns:
        the diff pattern.
    """
    if pattern1.shape != pattern2.shape:
        raise ValueError("patterns are not of equal shape")
    diffs = np.multiply(pattern1, pattern2)
    pattern_with_diffs = np.where(diffs < 0, diff_code, pattern1)
    return pattern_with_diffs


def flip_n(template, nr_of_flips):
    """
    makes a copy of the template pattern and flips
    exactly n randomly selected states.
    Args:
        template:
        nr_of_flips:
    Returns:
        a new pattern
    """
    n = np.prod(template.shape)
    # pick nrOfMutations indices (without replacement)
    idx_reassignment = np.random.choice(n, nr_of_flips, replace=False)
    linear_template = template.flatten()
    linear_template[idx_reassignment] = -linear_template[idx_reassignment]
    return linear_template.reshape(template.shape)


def get_noisy_copy(template, noise_level):
    """
    Creates a copy of the template pattern and reassigns N pixels. N is determined
    by the noise_level
    Note: reassigning a random value is not the same as flipping the state. This
    function reassigns a random value.

    Args:
        template:
        noise_level: a value in [0,1]. for 0, this returns a copy of the template.
        for 1, a random pattern of the same size as template is returned.
    Returns:

    """
    if noise_level == 0:
        return template.copy()
    if noise_level < 0 or noise_level > 1:
        raise ValueError("noise level is not in [0,1] but {}0".format(noise_level))
    linear_template = template.copy().flatten()
    n = np.prod(template.shape)
    nr_mutations = int(round(n * noise_level))
    idx_reassignment = np.random.choice(n, nr_mutations, replace=False)
    rand_values = np.random.binomial(1, 0.5, nr_mutations)
    rand_values = rand_values * 2 - 1  # map {0,1} to {-1, +1}
    linear_template[idx_reassignment] = rand_values
    return linear_template.reshape(template.shape)


def compute_overlap(pattern1, pattern2):
    """
    compute overlap

    Args:
        pattern1:
        pattern2:

    Returns: Overlap between pattern1 and pattern2

    """
    shape1 = pattern1.shape
    if shape1 != pattern2.shape:
        raise ValueError("patterns are not of equal shape")
    dot_prod = np.dot(pattern1.flatten(), pattern2.flatten())
    return float(dot_prod) / (np.prod(shape1))


def compute_overlap_list(reference_pattern, pattern_list):
    """
    Computes the overlap between the reference_pattern and each pattern
    in pattern_list

    Args:
        reference_pattern:
        pattern_list: list of patterns

    Returns:
        A list of the same length as pattern_list
    """
    overlap = np.zeros(len(pattern_list))
    for i in range(0, len(pattern_list)):
        overlap[i] = compute_overlap(reference_pattern, pattern_list[i])
    return overlap


def compute_overlap_matrix(pattern_list):
    """
    For each pattern, it computes the overlap to all other patterns.

    Args:
        pattern_list:

    Returns:
        the matrix m(i,k) = overlap(pattern_list[i], pattern_list[k]
    """
    nr_patterns = len(pattern_list)
    overlap = np.zeros((nr_patterns, nr_patterns))
    for i in range(nr_patterns):
        for k in range(i, nr_patterns):
            if i == k:
                overlap[i, i] = 1  # no need to compute the overlap with itself
            else:
                overlap[i, k] = compute_overlap(pattern_list[i], pattern_list[k])
                overlap[k, i] = overlap[i, k]  # because overlap is symmetric
    return overlap


def load_alphabet():
    """Load alphabet dict from the file
    ``data/alphabet.pickle.gz``, which is included in
    the neurodynex3 release.

    Returns:
        dict: Dictionary of 10x10 patterns

    Raises:
        ImportError: Raised if ``neurodynex``
            can not be imported. Please install
            `neurodynex <pypi.python.org/pypi/neurodynex/>`_.
    """
    # Todo: consider removing the zip file and explicitly store the strings here.
    file_str = "data/alphabet.pickle.gz"

    try:
        file_name = resource_filename("neurodynex3", file_str)
    except ImportError:
        raise ImportError(
            "Could not import data file %s. " % file_str +
            "Make sure the pypi package `neurodynex` is installed!"
        )

    with gzip.open("%s" % file_name) as f:
        if sys.version_info < (3, 0, 0):
            # python2 pickle.loads has no attribute "encoding"
            abc_dict = pickle.load(f)
        else:
            # latin1 is required for python3 compatibility
            abc_dict = pickle.load(f, encoding="latin1")

    # shape the patterns and provide upper case keys
    ABC_dict = dict()
    for key in abc_dict:
        ABC_dict[key.upper()] = abc_dict[key].reshape((10, 10))
    return ABC_dict


if __name__ == "__main__":
    pf = PatternFactory(5)
    L = pf.create_L_pattern(l_width=2)
    print(L)
