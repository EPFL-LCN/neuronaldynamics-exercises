"""
This file implements randomly two neuron models, as
NeuronX and NeuronY. One is of neurons.type1 and
one is of type neurons.type2

Relevant book chapters:

- http://neuronaldynamics.epfl.ch/online/Ch4.S4.html

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

from .neurons import NeuronTypeOne, NeuronTypeTwo
import numpy as np
import sys


def create_models():
    """Creates classes in this module that are random
    assignments of Type1 and Type2 neuron models."""

    print "Re-assigning Type1 and Type2 neuron models " \
        "randomly to classes NeuronX and NeuronY."

    # get a random assignment for X and Y
    order = np.random.permutation(2)
    suffixes = ['X', 'Y']
    thismodule = sys.modules[__name__]
    classes = [NeuronTypeOne, NeuronTypeTwo]

    # for each of these, assign classes in this module
    for i, o in enumerate(order):
        scname = 'Neuron%s' % suffixes[i]
        sc = type(scname, (classes[o],), {})
        setattr(thismodule, scname, sc)

create_models()
