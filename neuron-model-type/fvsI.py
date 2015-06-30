# This file is part of the exercise code repository accompanying 
# the book: Neuronal Dynamics (see <http://neuronaldynamics.epfl.ch>)
# located at <http://github.com/EPFL-LCN/neuronaldynamics-exercises>.
# 
# This free software: you can redistribute it and/or modify it under 
# the terms of the GNU General Public License 2.0 as published by the 
# Free Software Foundation. You should have received a copy of the 
# GNU General Public License along with the repository. If not, 
# see <http://www.gnu.org/licenses/>.
#
# Should you reuse and publish the code for your own purposes, 
# please cite the book or point to the webpage <http://neuronaldynamics.epfl.ch>.
# 
# Wulfram Gerstner, Werner M. Kistler, Richard Naud, and Liam Paninski. 
# Neuronal Dynamics: From Single Neurons to Networks and Models of Cognition. 
# Cambridge University Press, 2014.

# Simple script to plot an f vs I curve
# run in pylab using:
#
# execfile('fvsI.py')

import numpy
import sys
import tools
import TypeX
import TypeY
from matplotlib.pylab import *

I = numpy.arange(0.0,1.05,0.05)
f = []

string = "Plotting the firing rate of the neuron versus the amplitude"
string += "\r\n"
string += "of injecting step current. It may take several minutes ... "
string += "\r\n"

print string

for x in I:
    # watch progress
    print x
    sys.stdout.flush()
    f.append(tools.f(TypeX,x))

figure()
plot(I,f)
xlabel('Amplitude of Injecting step current (pA)')
ylabel('Firing rate (Hz)')
grid()
show()
