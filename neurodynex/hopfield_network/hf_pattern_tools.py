"""
Functions to create 2D patterns.
Note, in the hopfield model, we define patterns as vectors. To make
the exercise more visual, we use 2D patterns (N by N ndarrays).
"""
import numpy as np
from scipy import linalg
import pickle
import gzip
from pkg_resources import resource_filename
import sys


class PatternFactory:
    """
    Creates patterns of size N by N
    """
    def __init__(self, n):
        """
        Constructor
        Args:
            n: the width of a pattern

        Returns:

        """
        self.N = n

    def create_random_pattern(self, on_probability=0.5):
        p = np.random.binomial(1, on_probability, self.N ** 2)
        p = p*2 - 1
        return p.reshape((self.N, self.N))

    def create_random_pattern_list(self, nr_patterns, on_probability=0.5):
        p = list()
        for i in range(nr_patterns):
            p.append(self.create_random_pattern(on_probability))
        return p

    def create_all_on(self):
        return np.ones((self.N, self.N), np.int)

    def create_all_off(self):
        return -1*np.ones((self.N, self.N), np.int)

    def create_checkerboard(self):
        p = np.ones(self.N, np.int)
        # set every second value to -1
        p[1::2] = -1
        t = linalg.toeplitz(p)
        t = t.reshape((self.N, self.N))
        return t


def get_pattern_diff(pattern1, pattern2, diff_code=0):
    diffs = np.multiply(pattern1, pattern2)
    pattern_with_diffs = np.where(diffs < 0, diff_code, pattern1)
    return pattern_with_diffs

# def create3by3Cross():
#     """
#     creates the pattern used in one of the classroom exercises.
#     Returns:
#         3x3 cross
#     """
#     p = np.array([-1, 1, -1, 1, 1, 1, -1, 1, -1])
#     return p.reshape((3,3))


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
    n = template.shape[0]
    # pick nrOfMutations indices (without replacement)
    idx_reassignment = np.random.choice(n ** 2, nr_of_flips, replace=False)
    linear_template = template.flatten()
    linear_template[idx_reassignment] = -linear_template[idx_reassignment]
    return linear_template.reshape((n, n))


def get_noisy_copy(template, noise_level):
    n = template.shape[0]
    nr_mutations = int(round(n ** 2 * noise_level))
    if nr_mutations == 0:
        return template.copy()
    idx_reassignment = np.random.choice(n**2, nr_mutations, replace=False)
    linear_template = template.flatten()
    p = np.random.binomial(1, 0.5, n)
    p = p*2 - 1
    linear_template[idx_reassignment] = p
    return linear_template.reshape((n, n))


def compute_overlap(pattern1, pattern2):
    n = pattern1.shape[0]
    dot_prod = np.dot(pattern1.flatten(), pattern2.flatten())
    return float(dot_prod)/n**2


def compute_overlap_list(reference_pattern, pattern_list):
    overlap = np.zeros(len(pattern_list))
    for i in range(0, len(pattern_list)):
        overlap[i] = compute_overlap(reference_pattern, pattern_list[i])
    return overlap


def compute_overlap_matrix(pattern_list):
    nr_patterns = len(pattern_list)
    overlap = np.zeros((nr_patterns, nr_patterns))
    for i in range(nr_patterns):
        for k in range(i, nr_patterns):
            overlap[i, k] = compute_overlap(pattern_list[i], pattern_list[k])
            overlap[k, i] = overlap[i, k]  # because overlap is symmetric
    return overlap


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
            abc_dict = pickle.load(f)
        else:
            # latin1 is required for python3 compatibility
            abc_dict = pickle.load(f, encoding='latin1')

    for key in abc_dict:
        flat_pattern = abc_dict[key]
        abc_dict[key] = flat_pattern.reshape((10, 10))
    return abc_dict
