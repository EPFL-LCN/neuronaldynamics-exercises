# from brian2 import *
# import datetime
#
# # import os
# import matplotlib.pyplot as plt
#
#
# def fixed_indegree(indegree,n_post_pop,n_pre_pop):
#
#     presyn_indices = np.zeros([n_post_pop*indegree])
#     postsyn_indices = np.zeros([n_post_pop*indegree])
#     counter = 0
#
#     for post in range(n_post_pop):
#         x = np.arange(0, n_pre_pop)
#         y = np.random.permutation(x)
#         for i in range(indegree):
#             presyn_indices[counter] = y[i]
#             postsyn_indices[counter] = post
#             counter += 1
#     presyn_indices = presyn_indices.astype(int)
#     postsyn_indices = postsyn_indices.astype(int)
#     return presyn_indices, postsyn_indices
#
#
# def ISI_distribution(Spike_Trains,bin_length,t_start,t_end):# Spike trains is athe dictionary mapping neuon number with the corresponding spike times
#     # bin_length in ms
#     bins=(t_end-t_start)/bin_length
#     CV_ISI=np.zeros(len(Spike_Trains))
#     for i in range(len(Spike_Trains)):
#         temp1=np.diff(Spike_Trains[i]/ms)
#         temp2,_=np.histogram(temp1,bins)#,range=(t_start,t_end))
#         CV_ISI[i]=np.std(temp2)/np.mean(temp2)
#     mean_CV=np.mean(CV_ISI)
#     return CV_ISI, mean_CV
#
#
# def Brunel_Network(g,ro,N,C,w0,t_sim):
#
#     print "Start function", datetime.datetime.now()
#     delta_t=delta_t=defaultclock.dt
#     R = 20.0*Mohm
#     v_reset=0.0*mV
#     tau_m = 20.0*ms
#     tau_ref = 2.0*ms
#     v_th=20*mV
#     delta=1.5*ms
#
#     nu_ext=10*Hz
#
#     N_Exc=N
#     N_Inh=N/4
#     N_Poisson=N
#     CE=C
#     CI=C/4
#     Cpoisson=C
#     # Connection probability
#
#
#     wEE=w0*mV
#     wIE=w0*mV
#     wExt=w0*mV
#     wII=-g*w0*mV
#     wEI=-g*w0*mV
#     wp=0.1*mV
#     eqs = '''
#             dv/dt =-v/tau_m : volt
#           '''
#     Exc= NeuronGroup(N_Exc, model=eqs,reset='v=v_reset', threshold='v> v_th', refractory=tau_ref,)
#     Inh= NeuronGroup(N_Inh, model=eqs,reset='v=v_reset', threshold='v> v_th', refractory=tau_ref)
#     ######Ext=
#     print "making matrix", datetime.datetime.now()
#
#
#     #EE
#     presyn_indices,postsyn_indices=fixed_indegree(CE,N_Exc,N_Exc)
#     syn_EE= Synapses(Exc, Exc, on_pre='v += wEE', delay=delta)
#     syn_EE.connect(i=presyn_indices,j=postsyn_indices)
#
#     #IE
#     presyn_indices,postsyn_indices=fixed_indegree(CE,N_Inh,N_Exc)
#     syn_IE= Synapses(Exc, Inh, on_pre='v += wIE', delay=delta)
#     syn_IE.connect(i=presyn_indices,j=postsyn_indices)
#
#     #EI
#     presyn_indices,postsyn_indices=fixed_indegree(CI,N_Exc,N_Inh)
#     syn_EI= Synapses(Inh, Exc, on_pre='v +=wEI', delay=delta)
#     syn_EI.connect(i=presyn_indices,j=postsyn_indices)
#     #II
#     presyn_indices,postsyn_indices=fixed_indegree(CI,N_Inh,N_Inh)
#     syn_II= Synapses(Inh, Inh, on_pre='v +=wII', delay=delta)
#     syn_II.connect(i=presyn_indices,j=postsyn_indices)
#
#
#     #E_Iext
#     P = PoissonGroup(N_Poisson,ro*Hz)
#     presyn_indices,postsyn_indices=fixed_indegree(CE,N_Exc,N_Poisson)
#     syn_EP= Synapses(P,Exc, on_pre='v +=wp', delay=delta)
#     syn_EP.connect(i=presyn_indices,j=postsyn_indices)
#
#     presyn_indices,postsyn_indices=fixed_indegree(CE,N_Inh,N_Poisson)
#     syn_IP= Synapses(P, Inh, on_pre='v +=wp', delay=delta)
#     syn_IP.connect(i=presyn_indices,j=postsyn_indices)
#
#     #monitors
#     SpikeMonitor_Exc = SpikeMonitor(Exc)
#     SpikeMonitor_Inh=SpikeMonitor(Inh)
#     SpikeMonitor_Poisson=SpikeMonitor(P)
#
#
#     #Monitoring population rates
#     Pop_Rate_Exc = PopulationRateMonitor(Exc)
#     Pop_Rate_Inh = PopulationRateMonitor(Inh)
#
#
#     '''
#     #Monitoring input currents to neurons # can only record from a few to avoid memory shortage
#     Input_Exc= StateMonitor(Exc, 'v', record=[1,2,3])
#     Input_Inh=StateMonitor(Inh, 'v', record=[1,2,3])
#     '''
#
#
#     #initialization
#     Exc.v=0.0*mV
#     Inh.v=0.0*mV
#
#     print "before simulation", datetime.datetime.now()
#
#     run(t_sim*ms, report='text',report_period=10*second)
#
#     print "after simulation", datetime.datetime.now()
#
#
#     colors = {}
#     colors['Exc'] = 'blue'
#     colors['Inh'] = 'red'
#     colors['Mean'] = 'green'
#
#     print "it's done!", datetime.datetime.now()
#
#
#     SpikeTrain_Exc=SpikeMonitor_Exc.spike_trains()
#     SpikeTrain_Inh=SpikeMonitor_Inh.spike_trains()
#
#
#     #figure()
#     #plot(SpikeMonitor_Exc.t/ms, SpikeMonitor_Exc.i, '.',label='Excitatory neurons',color=colors['Exc'])
#     #plot(SpikeMonitor_Poisson.t/ms, SpikeMonitor_Poisson.i, '.',label='Poisson')
#     figure()
#     for i in range(50):
#         plot(SpikeTrain_Exc[i]/ms, (i+1)*np.ones(len(SpikeTrain_Exc[i])), '.',color=colors['Exc'])
#     #plot(M_Inh.t/ms, N_Exc+M_Inh.i, '.',label='Inhibitory neurons',color=colors['Inh'])
#     xlabel('Time (ms)')
#     ylabel(' Neuron #')
#     legend()
#
#
#
#     _,CV_Exc=ISI_distribution(SpikeTrain_Exc,1,100,t_sim)
#     _,CV_Inh=ISI_distribution(SpikeTrain_Inh,1,100,t_sim)
#     CV_mean=np.mean([CV_Exc,CV_Inh])
#     print CV_Exc,CV_Inh
#     print "CV of ISI is:", CV_mean
#     Mean_Pop_Rate=(Pop_Rate_Exc.smooth_rate(window='flat', width=0.1*ms)*N_Exc+Pop_Rate_Inh.smooth_rate(window='flat', width=0.1*ms)*N_Inh)/(N_Exc+N_Inh)
#     Mean_Pop_Activity=np.mean(Mean_Pop_Rate[200*ms/delta_t:])
#     Std_Pop_Activity=np.std(Mean_Pop_Rate[200*ms/delta_t:])
#     print Mean_Pop_Activity
#     print Std_Pop_Activity
#     print "CV of population activity (A(t)) is:",Std_Pop_Activity/Mean_Pop_Activity
#
#     figure()
#     #plot(Pop_Rate_Exc.t/ms, Pop_Rate_Exc.smooth_rate(window='flat', width=5*ms)/Hz,linewidth=1,label='Exc',color=colors['Exc'])
#     #plot(Pop_Rate_Inh.t/ms, Pop_Rate_Inh.smooth_rate(window='flat', width=5*ms)/Hz,linewidth=1, label='Inh',color=colors['Inh'])
#     plot(Pop_Rate_Inh.t/ms, Mean_Pop_Rate/Hz,linewidth=1, label='Mean population activity',color=colors['Mean'])
#     plt.axhline(y=(-Std_Pop_Activity+Mean_Pop_Activity)/Hz,c='green',ls='--',linewidth=1)
#     plt.axhline(y=(Mean_Pop_Activity+Std_Pop_Activity)/Hz,c='green',ls='--',linewidth=1)
#     plt.axhline(y=(Mean_Pop_Activity)/Hz,c='green',ls='--',linewidth=2)
#     #plot(Pop_Rate_Inh.t/ms, Mean_Pop_Rate/Hz,linewidth=1, label='Mean population activity',color=colors['Mean'])
#     xlabel('Time (ms)')
#     xlim( xmin=50)
#     ylabel('Population activity (Hz)')
#     legend()
#     show()
#
# #Brunel_Network(4,16,5000,500,0.0,4000)  #g,ro,N,C,w0,t_sim # Why rate of Inh is higher compared to Exc even when w=0.0?!!
# #Brunel_Network(3,15,10000,1000,0.1,300) #Top left, g=3, rate=30-->g=3, rate=16, SR
# #Brunel_Network(5,16,10000,1000,0.1,300) #bottom left, g=5, rate=16 #not bad but not totally AI!!
# #Brunel_Network(6,32,10000,1000,0.1,150) #seems to be a better Bottom Left, i.e. AI!!(still need to measure regularity with ISI)
# #Brunel_Network(6,32,10000,1000,0.1,400) #bottom left, g=5, rate=16
# #Brunel_Network(6,70,10000,1000,0.1,300) #bottom left, g=5, rate=16 g=6,rate=70-->almost SI!! (if I increase the weight it seems to work better)
# Brunel_Network(4.5,12,10000,1000,0.1,300)##bottom right, g=4.5 rate=12, SI slow
#
