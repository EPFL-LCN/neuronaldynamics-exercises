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

import numpy

def spiketimes(t,v,v_th = 0.5):
    """Given voltage and time, returns array of spike times

    INPUT:

    t - time
    v - voltage, same shape as t

    """

    v_above_th = v>v_th
    idx = numpy.nonzero((v_above_th[:-1]==False)&(v_above_th[1:]==True))
    return t[idx[0]+1]
    
def f(TypeXY, I):
    """Return the firing rate of neuron TypeXY under input current I

    INPUT:

    TypeXY - Neuron model (with a Step function)
    I - input current in amp.

    """

    t,v,w,I = TypeXY.Step(I_amp=I,Step_tstart=100,Step_tend=1000,tend=1000)

    st = spiketimes(t,v)

    # no spikes or 1 spike
    if len(st)<2:
        return 0.0

    isi = st[1:]-st[:-1]

    f = 1000.0/numpy.mean(isi)

    return f


