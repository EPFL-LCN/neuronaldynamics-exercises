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

from brian2 import *
from brian2.units.stdunits import nA
import matplotlib.pyplot as plt
import brian2
import numpy


def LIF_Step(I_tstart = 20, I_tend = 70, I_amp=1.005, tend=100):
      
    """
Run the LIF and give a step current input.

Parameters:
tend = 100    (ms) - is the end time of the model    
I_tstart = 20 (ms) - start of current step
I_tend = 70   (ms) - end of current step
I_amp = 1.005 (nA) - amplitude of current step
    """
    
    # neuron parameters
    R = 1*Mohm
    v_reset = '''v=0*mV'''
    v_rest = '''0*mV'''
    v_threshold = '''v>1*mV'''
    tau = 1*ms
    
    # producing step current
    Step_current = numpy.zeros(tend)
    for t in range(tend):
        if(I_tstart <= t and t <= I_tend):
            # Step_current[t] = I_amp* 0.000000001 * amp
            Step_current[t] = I_amp * 1e-9 * amp
    print Step_current
    
    # differential equation of Leaky Integrate-and-Fire model    
    eqs = '''
    dv/dt = (-(v-v_rest) + R*I(t))/(tau) : volt
    I : amp
    # I : amp
    '''
    # making neuron using Brian library
    I = TimedArray(Step_current,dt=1*ms)
    IF = NeuronGroup(1, model=eqs,reset=v_reset,threshold=v_threshold)
    IF.v = v_rest

    
    # monitoring membrane potential of neuron and injecting current
    Mv = StateMonitor(IF, 'v', record=True)
    Current = StateMonitor(IF, 'I', record=True)
    
    # initialization of simulator
    #reinit()
    
    # run the simulation
    run(tend * ms)
        
    # plotting figures
    subplot(211)
    plot(Mv.times/ms, Mv[0]/mV,lw=2)
    plot([0,tend],[v_threshold/mV,v_threshold/mV],'r--',lw=2)
    xlabel('t')
    ylabel('v')
    ylim(0,(v_threshold/mV)*1.2)
    grid()
    
    subplot(212)
    plot(Current.times/ms, Current[0]/nA,lw=2)
    xlabel('t')
    ylabel('I')
    grid()
    
    show()
    

def LIF_Sinus(I_freq = 0.1,I_offset=0.5,I_amp=0.5, tend=100):
    
    """
Run the LIF for a sinusoidal current

Parameters:
tend = 100     (ms) - is the end time of the model    
I_freq = 0.1   (kHz) - frequency of current sinusoidal
I_offset = 0.5 (nA) - offset current
I_amp = 0.5    (nA) - amplitude of sin
    """
    
    # setting neuron parameters
    R = 1*Mohm
    v_reset = 0*mV
    v_rest = 0*mV
    v_threshold = 1*mV
    tau = 1*ms
    
    # producing sinusoidal current
    t = numpy.arange(0,tend,0.1)
    Current = I_amp*numpy.sin(2.0*numpy.pi*I_freq*t)+I_offset

    # differential equation of Leaky Integrate-and-Fire model    
    eqs = '''
    dv/dt = (-(v-v_rest) + R*I)/(tau) : volt
    I : nA
    '''
    # making neuron using Brian library
    IF = NeuronGroup(1, model=eqs,reset=v_reset,threshold=v_threshold)
    IF.v = v_rest
    IF.I = TimedArray(Current*nA,dt=0.1*ms)
    
    # monitoring membrane potential of neuron and injecting current
    Mv = StateMonitor(IF, 'v', record=True)
    Current = StateMonitor(IF, 'I', record=True)
    
    # initialization of simulator
    reinit()
    
    # run the simulation
    run(tend*ms)
    
    # plotting figures
    figure()
    subplot(211)
    plot(Mv.times/ms, Mv[0]/mV,lw=2)
    plot([0,tend],[v_threshold/mV,v_threshold/mV],'r--',lw=2)
    xlabel('t(ms)')
    ylabel('V(mV)')
    ylim(0,(v_threshold/mV)*1.2)
    grid()
    
    subplot(212)
    plot(Current.times/ms, Current[0]/nA,lw=2)
    xlabel('t(ms)')
    ylabel('I(nA)')
    grid()
    
    show()
   
    
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
string += "This file implements leaky intergrate-and-fire(LIF) model."
string += "\r\n"
string += "You can inject a step current or sinusoidal current into" 
string += "\r\n"
string += "neuron using LIF_Step() or LIF_Sinus() methods respectively."
string += "\r\n"
string += "\r\n"
string += "In order to know parameters and default values for each method"
string += "\r\n"
string += "use symbol ? after the name of method. For example: LIF_Step?"
string += "\r\n\r\n"
string += "-------------2014 EPFL-LCN all rights reserved----------------"
string += "\r\n"

print string
