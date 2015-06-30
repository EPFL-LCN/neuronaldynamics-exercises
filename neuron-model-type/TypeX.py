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
 
from brian import *
from pylab import *
import numpy

def Step(tend=300, Step_tstart = 20, Step_tend = 270, I_amp=.5):
    """
    DEFINITION
    Runs the TypeX model for a step current.
    
    INPUT
    tend: ending time (ms)
    Step_tstart: time at which the step current begins (ms)
    Step_tend: time  at which the step current ends (ms)
    I_amp: magnitude of the current step (uA/cm^2)
    
    OUTPUT

    [t,v,w,I]
    
    t - time
    v - voltage trace
    w - slow recovery variable
    I - current injected
    """

    # producing step current
    Step_current = numpy.zeros(tend)
    for t in range(tend):
        if(Step_tstart <= t and t <= Step_tend):
            Step_current[t] = I_amp*pA
    
    # converting to acceptable current for Brian
    I = TimedArray(Step_current,dt=1*ms)
    
    [my_t, v, w, I] = model(tend,I)

    return [my_t/ms,v/mV,w/mV,I/pA]

def PlotStep(tend=300, Step_tstart = 20, Step_tend = 270, I_amp=.5):
    """
    
    DEFINITION
    Plots the TypeX model for a step current.
    
    INPUT
    tend: ending time (ms)
    Step_tstart: time at which the step current begins (ms)
    Step_tend: time  at which the step current ends (ms)
    I_amp: magnitude of the current step (uA/cm^2)
    
    OUTPUT
    graph with three panels: 1) voltage trace, 2) slow recovery variable w 3) current injected.

    """

    t,v,w,I = Step(tend, Step_tstart, Step_tend, I_amp)

    # open new figure and plot
    # plot voltage time series
    subplot(311)
    plot(t,v,lw=2)
    xlabel('t (ms)')
    ylabel('v (mv)')
    grid()
    
    # plot activation and inactivation variables
    subplot(312)
    plot(t,w,'k',lw=2)
    xlabel('t (ms)')
    ylabel('w (mv)')
    grid()
    
    # plot current
    subplot(313)
    plot(t,I,lw=2)
    axis((0, tend, 0, I_amp*1.1))
    xlabel('t (ms)')
    ylabel('I (pA)')
    grid()
    
    show()
    
def model(tend,I):
    
    # neuron parameters
    a = 1.25
    tau = 15.6 * ms
     
    # forming the neuron model using differential equations
    eqs = '''
    I_e : amp  
    dv/dt = (v - (v**3)/(3*mvolt*mvolt) - w + I_e*Gohm)/ms : volt
    dw/dt = (a*(v+0.7*mvolt)-w)/tau : volt
    '''
    
    neuron = NeuronGroup(1, eqs)
    
    # initialization of simulator
    reinit()
    
    # parameter initialization
    neuron.v = 0
    
    # injecting current to the neuron
    neuron.I_e = I
    
    # tracking parameters
    tracev = StateMonitor(neuron, 'v', record=True)
    tracew = StateMonitor(neuron, 'w', record=True)
    traceI = StateMonitor(neuron, 'I_e', record=True)

    # running the simulation
    run(tend * ms)

    return [tracev.times, tracev[0], tracew[0], traceI[0]]


string = "***********************************************************"
string += "\r\n"
string += "***** Ecole polytechnique federale de Lausanne (EPFL) *****"
string += "\r\n"
string += "***** Laboratory of computational neuroscience (LCN)  *****"
string += "\r\n"
string += "*****               Neuronal Dynamics                 *****"
string += "\r\n"
string += "***********************************************************"
string += "\r\n"
string += "This file implements a two dimensional neuron model. You can"
string += "\r\n"
string += "inject a step current into it using Step() method, but it does" 
string += "\r\n"
string += "not show the figure. Instead, you can use method PlotStep() for"
string += "\r\n"
string += "both injecting current and observing the figure."
string += "\r\n"
string += "\r\n"
string += "In order to know parameters and default values for each method"
string += "\r\n"
string += "use symbol ? after the name of method. For example: Step?"
string += "\r\n\r\n"
string += "-------------2014 EPFL-LCN----------------"
string += "\r\n"

print string
