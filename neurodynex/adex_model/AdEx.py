# from brian import *
# import matplotlib.pyplot as plt
#
# defaultclock.dt=0.01*ms
# # This function implement Adaptive Exponential Leaky Integrate-And-Fire neuron model
# def AdEx_model(C,gL,EL,VT,DeltaT,a,tauw,b,Vr,I_amp):
#
#     #Implementing AdEx model using the following differential equations:
#     eqs='''
#         dvm/dt=(gL*(EL-vm)+gL*DeltaT*exp((vm-VT)/DeltaT)+Ie-w)/C : volt
#         dw/dt=(a*(vm-EL)-w)/tauw : amp
#         Ie: amp
#         '''
#     Vcut=0.0 *mV # practical threshold condition
#     neuron=NeuronGroup(1,model=eqs,threshold=Vcut,reset="vm=Vr;w+=b")
#     reinit()
#
#     #initial values of vm and w is set here:
#     neuron.vm=EL
#     neuron.w=0.0*pA
#     #amplitude of step current
#     neuron.Ie=I_amp
#
#     #Monitoring membrane voltage (vm) and w!
#     voltage= StateMonitor(neuron, 'vm',timestep=1, record=True)
#     adaptation= StateMonitor(neuron, 'w',timestep=1, record=True)
#
#     #running simulation for 500 ms
#     run(500*ms)
#
#     #Plotting membrane voltage as a function of time, and phase plane representation of vm and w.
#     plt.subplot(2,2,1)
#     plt.plot(voltage.times/ms,voltage[0]/mV,lw=2)
#     xlabel('t')
#     ylabel('Vm')
#     title('Membrane potential')
#     plt.subplot(2,2,2)
#     plt.plot(voltage[0]/mV,adaptation[0]/pA,lw=2)
#     plt.xlabel('Vm')
#     plt.ylabel('W')
#     plt.title('Phase plane representation')
#     plt.subplot(2,2,3)
#     plt.plot(adaptation.times/ms,adaptation[0]/pA,lw=2)
#     plt.xlabel('t')
#     plt.ylabel('W')
#     plt.title('Adaptation current')
#
#     plt.show()
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
# string += "This file implements Adaptive Exponential Leaky Intergrate-And-Fire (`AdEx) model."
# string += "\r\n"
# string += "You can generate various firing patterns by assigning appropriate values to the parameters"
# string += "\r\n"
# string += "specifically a, b, and tau_w"
# string += "\r\n\r\n"
# string += "-------------2015 EPFL-LCN all rights reserved----------------"
# string += "\r\n"
#
# print string
#
#
#
# # Here we call the function, using appropriate values of parameters to generate different firing patterns.
# AdEx_model(200.0*pF,12.0*nS,-70.0*mV,-50.0*mV,2.0*mV,2.0*nS,300.0*ms,60.0*pA,-58.0*mV,500.0*pA)
#
# #tonic spiking
# #C,gL,EL,VT,DeltaT,a,tauw,b,Vr,I_amp=200.0*pF,10.0*nS,-70.0*mV,-50.0*mV,10.0*mV,2.0*nS,30.0*ms,0.0*pA,-58.0*mV,500.0*pA#DeltaT=10*mV
# #adaptation
# #C,gL,EL,VT,DeltaT,a,tauw,b,Vr,I_amp=200.0*pF,12.0*nS,-70.0*mV,-50.0*mV,2.0*mV,2.0*nS,300.0*ms,60.0*pA,-58.0*mV,500.0*pA #DeltaT=5*mV reaches the threshold better
# #initial burst
# #C,gL,EL,VT,DeltaT,a,tauw,b,Vr,I_amp=130.0*pF,18.0*nS,-58.0*mV,-50.0*mV,2.0*mV,4.0*nS,150.0*ms,120.0*pA,-50.0*mV,400.0*pA#DeltaT=10*mV
# #regular bursting
# #C,gL,EL,VT,DeltaT,a,tauw,b,Vr,I_amp=200.0*pF,10.0*nS,-58.0*mV,-50.0*mV,2.0*mV,2.0*nS,120.0*ms,100.0*pA,-46.0*mV,210.0*pA
# #delayed accelerating
# #C,gL,EL,VT,DeltaT,a,tauw,b,Vr,I_amp=200.0*pF,12.0*nS,-70.0*mV,-50.0*mV,2.0*mV,-10.0*nS,300.0*ms,0.0*pA,-58.0*mV,300.0*pA
# #delayed regular bursting #is not working, don't know why??? according to the book, maybe b=0! is a wrong assumption
# #C,gL,EL,VT,DeltaT,a,tauw,b,Vr,I_amp=200.0*pF,12.0*nS,-70.0*mV,-50.0*mV,2.0*mV,-6.0*nS,300.0*ms,0.0*pA,-58.0*mV,110.0*pA
# #transient spiking #is not working, don't know why?
# #C,gL,EL,VT,DeltaT,a,tauw,b,Vr,I_amp=100.0*pF,10.0*nS,-65.0*mV,-50.0*mV,2.0*mV,-10.0*nS,90.0*ms,30.0*pA,-50.0*mV,350.0*pA
# #irregular spiking
# #C,gL,EL,VT,DeltaT,a,tauw,b,Vr,I_amp=100.0*pF,12.0*nS,-60.0*mV,-50.0*mV,2.0*mV,-11.0*nS,130.0*ms,30.0*pA,-48.0*mV,160.0*pA