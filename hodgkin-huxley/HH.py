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
from brian.library.ionic_currents import *
import numpy



def HH_model(tend,I):
    
    # neuron parameters
    El = 10.6 * mV
    EK = -12 * mV
    ENa = 115 * mV
    gl = 0.3 * msiemens
    gK = 36 * msiemens
    gNa = 120 * msiemens
    C = 1 * ufarad
    
    # forming HH model with differential equations
    eqs = '''
    I_e : amp  
    __membrane_Im = I_e+ gNa*m**3*h*(ENa-vm) + gl*(El-vm) + gK*n**4*(EK-vm) : amp    
    alphah = .07*exp(-.05*vm/mV)/ms    : Hz
    alpham = .1*(25*mV-vm)/(exp(2.5-.1*vm/mV)-1)/mV/ms : Hz    
    alphan = .01*(10*mV-vm)/(exp(1-.1*vm/mV)-1)/mV/ms : Hz    
    betah = 1./(1+exp(3.-.1*vm/mV))/ms : Hz    
    betam = 4*exp(-.0556*vm/mV)/ms : Hz    
    betan = .125*exp(-.0125*vm/mV)/ms : Hz    
    dh/dt = alphah*(1-h)-betah*h : 1    
    dm/dt = alpham*(1-m)-betam*m : 1    
    dn/dt = alphan*(1-n)-betan*n : 1    
    dvm/dt = __membrane_Im/C : volt    
    '''
    
    neuron = NeuronGroup(1, eqs, implicit=True, freeze=True)
    
    # initialization of simulator
    reinit()
    
    # parameter initialization
    neuron.vm = 0
    neuron.m = 0.0529324852572
    neuron.h = 0.596120753508
    neuron.n = 0.317676914061
    
    # injecting current to the neuron
    neuron.I_e = I
    
    # tracking parameters
    tracev = StateMonitor(neuron, 'vm', record=True)
    traceI = StateMonitor(neuron, 'I_e', record=True)
    tracem = StateMonitor(neuron, 'm', record=True)
    tracen = StateMonitor(neuron, 'n', record=True)
    traceh = StateMonitor(neuron, 'h', record=True)
    
    # running the simulation
    run(tend * ms)
    
    # plot voltage time series
    subplot(311)
    plot(tracev.times / ms, tracev[0] / mV, lw=2)
    xlabel('t (ms)')
    ylabel('v (mV)')
    grid()
    
    # find max of activation and inactivation variables
    traceall = numpy.append(tracem[0],[tracen[0],traceh[0]])
    nrmfactor = numpy.max(traceall)/mV
    
    # plot activation and inactivation variables
    subplot(312)
    plot(tracev.times / ms, (tracem[0]/nrmfactor) / mV,'black', lw=2)
    plot(tracev.times / ms, (tracen[0]/nrmfactor) / mV,'blue', lw=2)
    plot(tracev.times / ms, (traceh[0]/nrmfactor) / mV,'red', lw=2)
    xlabel('t (ms)')
    ylabel('act./inact.')
    legend(('m','n','h'))
    grid()
    
    # plot current
    subplot(313)
    plot(traceI.times/ms,traceI[0]/uA,lw=2)
    axis((0, tend, min(traceI[0]/uA)*1.1, max(traceI[0]/uA)*1.1))
    xlabel('t (ms)')
    ylabel('I (micro A)')
    grid()
    
    show()
    
def HH_Step(Step_tstart = 20, Step_tend = 180, I_amp=7, tend=200):
    
    """
run the Hodgkin-Huxley for a step current

Parameters:
tend = 100           ending time (ms)
Step_tstart = 20     time at which the step current begins (ms)
Step_tend = 70       time at which the step current ends (ms)
I_amp = 7            magnitude of the current step (micro A)
    """
    
    # producing step current
    Step_current = numpy.zeros(tend)
    for t in range(tend):
        if(Step_tstart <= t and t <= Step_tend):
            Step_current[t] = I_amp*uA
    
    # converting to acceptable current for Brian
    I = TimedArray(Step_current,dt=1*ms)
    
    HH_model(tend,I)
        

def HH_Sinus(I_freq = 0.01, I_offset=0.5,I_amp=7, tend=600):
    """
Runs the Hodgkin Huxley model for a step current.
    
Parameters:
tend = 600         ending time (ms)
I_freq = 0.01      frequency of stimulating current (kHz)
I_offset = 0.5     offset (uA)
I_amp = 7          amplitude of the sin (micro A)
    """
    
    # producing sinusoidal current
    t = numpy.arange(0,tend,0.1)
    Current = I_amp*numpy.sin(2.0*numpy.pi*I_freq*t)+I_offset
    Current = Current * uA
    
    # converting to acceptable current for Brian
    I = TimedArray(Current,dt=0.1*ms)
        
    HH_model(tend,I)


def HH_Ramp(Ramp_tstart = 30, Ramp_tend = 270, FinalAmp=20, tend=300):
    """
Run the Hodgkin-Huxley model for a ramp current.

Parameters:
tend = 300        ending time (ms)
Ramp_tstart = 30  time at which the ramp current begins (ms)
Ramp_tend = 270   at which the ramp current ends (ms)
FinalAmp = 20     magnitude of the current at the end of the ramp (micro A)
    """
    # producing Ramp current
    t = numpy.arange(0,tend,0.01)  
    Current = numpy.zeros(t.shape)

    index_start = numpy.searchsorted(t,Ramp_tstart)
    index_end = numpy.searchsorted(t,Ramp_tend)
    Current[index_start:index_end] = numpy.arange(0,index_end-index_start,1.0)/(index_end-index_start)*FinalAmp
    Current = Current * uA
    
    # converting to acceptable current for Brian
    I = TimedArray(Current,dt=0.01*ms)
    
    HH_model(tend,I) 
    
string = "  ***********************************************************"
string += "\r\n"
string += "  ***** Ecole polytechnique federale de Lausanne (EPFL) *****"
string += "\r\n"
string += "  ***** Laboratory of computational neuroscience (LCN)  *****"
string += "\r\n"
string += "*****                 Neuronal Dynamics                 *****"
string += "\r\n"
string += "  ***********************************************************"
string += "\r\n"
string += "This file implements Hodgkin-Huxley (HH) model. You can inject"
string += "\r\n"
string += "a step current, sinusoidal current or ramp current into neuron" 
string += "\r\n"
string += "using HH_Step(), HH_Sinus() or HH_Ramp() methods respectively."
string += "\r\n"
string += "\r\n"
string += "In order to know parameters and default values for each method"
string += "\r\n"
string += "use symbol ? after the name of method. For example: HH_Step?"
string += "\r\n\r\n"
string += "-------------2014 EPFL-LCN all rights reserved----------------"
string += "\r\n"


print string
