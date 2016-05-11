# from brian import *
# defaultclock.dt = 0.1*ms
# import matplotlib.pyplot as plt
# import brian
# import numpy
# import math as m
#
# def expLIF_model(I,tend):
#
#     # neuron parameters
#     R = 20.0*Mohm
#     v_rest = 0.0*mV
#     v_reset=0.0*mV
#     theta_reset = 15.0*mV #15
#     v_rh=10.0*mV#12
#     delta_T=0.8*mV #0.5 mV and 0.1mV
#     tau = 30.0*ms
#     # differential equation of Exponential Leaky Integrate-and-Fire model
#     eqs = '''
#     dv/dt = (-(v-v_rest) +delta_T*exp((v-v_rh)/delta_T)+ R*Ie)/(tau) : mV
#     Ie : nA
#     '''
#
#     # making neuron using Brian library
#     IF = NeuronGroup(1, model=eqs,reset=v_reset, threshold=theta_reset) #v_reset instead of 0*mV
#     reinit()
#     # initialization of simulator
#     IF.v = v_rest
#     IF.Ie=I
#     # monitoring membrane potential of neuron and injecting current
#     Mv = StateMonitor(IF, 'v',timestep=1, record=True)
#     Current = StateMonitor(IF, 'Ie',timestep=1, record=True)
#     # run the simulation
#     print 'before run'
#     run(tend*ms)
#
#     # plotting figures
#     subplot(211)
#     plot(Mv.times/ms, Mv[0]/mV,lw=2)
#     #plot([0,tend],[v_reset/mV,v_reset/mV],'r--',lw=2)
#     plot([0,tend],[theta_reset/mV,theta_reset/mV],'r--',lw=2)
#     xlabel('t')
#     ylabel('v')
#     #ylim(0,(v_reset/mV)*1.2)
#     #ylim(0,15) # should be removed
#     grid()
#
#     subplot(212)
#
#     plot(Current.times/ms, Current[0]/nA,lw=2)
#     #plot(MYT.times/ms, MYTT[0],lw=2)
#     xlabel('t')
#     ylabel('I')
#     grid()
#
#
# def expLIF_Step(I_tstart = 20, I_tend = 200, I_amp=1.005, tend=300):
#
#     """
# Run the exponential LIF model and give a step current input.
#
# Parameters:
# tend = 300    (ms) - is the end time of the model
# I_tstart = 20 (ms) - start of current step
# I_tend = 200   (ms) - end of current step
# I_amp = 1.005 (nA) - amplitude of current step
#     """
#     # producing step current
#     Step_current = numpy.zeros(tend)
#     for t in range(tend):
#         if(I_tstart <= t and t <= I_tend):
#             Step_current[t] = I_amp*nA
#     I = TimedArray(Step_current,dt=1*ms)
#     expLIF_model(I,tend)
#
#
#
#
# def expLIF_Ramp(Ramp_tstart = 20, Ramp_tend = 200, FinalAmp=20, tend=300):
#     """
# Run the exponential LIF model for a ramp current.
#
# Parameters:
# tend = 300        ending time (ms)
# Ramp_tstart = 30  time at which the ramp current begins (ms)
# Ramp_tend = 270   at which the ramp current ends (ms)
# FinalAmp = 20     magnitude of the current at the end of the ramp (micro A)
#     """
#     # producing Ramp current
#     t = numpy.arange(0,tend,0.01)
#     Current = numpy.zeros(t.shape)
#
#     index_start = numpy.searchsorted(t,Ramp_tstart)
#     index_end = numpy.searchsorted(t,Ramp_tend)
#     Current[index_start:index_end] = numpy.arange(0,index_end-index_start,1.0)/(index_end-index_start)*FinalAmp
#     Current = Current * nA
#
#     # converting to acceptable current for Brian
#     I = TimedArray(Current,dt=0.01*ms)
#
#     expLIF_model(I, tend)
#
# def expLIF_pulse(Pulse_tstart=30,Pulse_tend=32,tend=300,PulseAmp=1.05):
#     """
# Run the exponential LIF model and give a step current input.
#
# Parameters:
# tend = 300    (ms) - is the end time of the model
# Pulse_tstart = 30 (ms) - start of current step
# Pulse_tend = 32   (ms) - end of current step
# I_amp = 1.005 (nA) - amplitude of current step
#     """
#     # producing step current
#     Step_current = numpy.zeros(tend)
#     for t in range(tend):
#         if(Pulse_tstart <= t and t <= Pulse_tend):
#             Step_current[t] = PulseAmp*nA
#     I = TimedArray(Step_current,dt=1*ms)
#     expLIF_model(I,tend)
#
#
#
#
# string = "***********************************************************"
# string += "\r\n"
# string += "***** Ecole polytechnique federale de Lausanne (EPFL) *****"
# string += "\r\n"
# string += "***** Laboratory of computational neuroscience (LCN)  *****"
# string += "\r\n"
# string += "*****     Biological modeling of neural networks      *****"
# string += "\r\n"
# string += "***********************************************************"
# string += "\r\n"
# string += "This file implements exponential leaky intergrate-and-fire(LIF) model."
# string += "\r\n"
# string += "You can inject a step current, ramp current, or short current pulse into"
# string += "\r\n"
# string += "neuron using expLIF_Step(), expLIF_Ramp(), or expLIF_pulse() methods respectively."
# string += "\r\n"
# string += "\r\n"
# string += "In order to know parameters and default values for each method"
# string += "\r\n"
# string += "use symbol ? after the name of method. For example: LIF_Step?"
# string += "\r\n\r\n"
# string += "-------------2013 EPFL-LCN all rights reserved----------------"
# string += "\r\n"
#
# print string
#
# #code for measurements for step input with different amplitudes.
# '''
# IAmp=numpy.arange(0.1,1.0,0.1)
# for i in IAmp:
#     expLIF_Step(I_tstart = 20, I_tend = 200, I_amp=i, tend=300)
# show()
# '''
#
#
#
# #code for measurments for ramp input with different slopes and a fixed max amplitude.
# '''
# ramp_tend=numpy.arange(30.0,300.0,50.0) #220.0-270.0,10.0
# for i in ramp_tend:
#     expLIF_Ramp(Ramp_tstart = 30, Ramp_tend =i, FinalAmp=2.0, tend=300)
# show()
# '''
#
#
#
#
# #code for measurements for pulse input with different amplitudes
#
# pulse_amp=numpy.arange(5.0,7.0,0.5)
# for i in pulse_amp:
#     expLIF_pulse(Pulse_tstart=30,Pulse_tend=31,PulseAmp=i,tend=300)
# show()
#
#
#
#
#
#
