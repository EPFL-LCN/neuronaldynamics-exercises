"""This file is part of the exercise code repository accompanying
the book: Neuronal Dynamics (see <http://neuronaldynamics.epfl.ch>)
located at <http://github.com/EPFL-LCN/neuronaldynamics-exercises>.

This free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License 2.0 as published by the
Free Software Foundation. You should have received a copy of the
GNU General Public License along with the repository. If not,
see <http://www.gnu.org/licenses/>.

Should you reuse and publish the code for your own purposes,
please cite the book or point to the webpage <http://neuronaldynamics.epfl.ch>.

Wulfram Gerstner, Werner M. Kistler, Richard Naud, and Liam Paninski.
Neuronal Dynamics: From Single Neurons to Networks and Models of Cognition.
Cambridge University Press, 2014."""

import unittest
# disable plotting output hopefully
import matplotlib
matplotlib.use('Agg')

if __name__ == '__main__':

    loader = unittest.TestLoader()
    tests = loader.discover('.')
    testRunner = unittest.runner.TextTestRunner(verbosity=2)
    testRunner.run(tests)
