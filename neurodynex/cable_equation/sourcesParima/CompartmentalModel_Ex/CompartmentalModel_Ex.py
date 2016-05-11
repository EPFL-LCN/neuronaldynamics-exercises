#Author: Parima Ahmadipour
#Copyright: LCN-2016
from brian2 import *
defaultclock.dt = 0.01*ms

#This function plots spatial and temporal evolution of membrane potential of a dendrite with N compartments, 
#simulated over t_sim ms and stimulated from its middle with a short pulse of I0 nA amplitude.
def SpatialAndTemporal_PotentialEvolution(N=200,I0=1,t_sim=5):
    
    # morphological and electrical parameters
    d=1*um #diameter of dendrite
    RL=100*ohm*cm #Intracellular medium resistance
    RT=30000*ohm*cm**2#cell mebrane resistance
    L=1000*um #length of dendrite
    EL=-70*mV #reversal potential
    tau=30*ms
    C=1*uF/cm**2
    
    # a dendrite with diameter=d
    my_morphology=Cylinder(diameter=d,length=L,n=N)
    
    #Im is transmembrane current
    #Iext is  injected current at a specific position on dendrite
    my_equs='''
    Iext: amp (point current)
    Im=(EL-v)/RT : amp/meter**2
    '''
    my_neuron=SpatialNeuron(morphology=my_morphology,model=my_equs,Cm=C,Ri=RL)
    potentials=StateMonitor(my_neuron,'v',record=True)
    my_neuron.Iext[N/2]=I0*nA # Injecting a short pulse current in the middleof dendrite
    my_neuron.v=EL+0.5*mV
    run(0.05*ms)
    my_neuron.Iext=0.0*nA
    run(t_sim*ms)
    
    figure()
    
    for i in range(int(N/10)):
        plot(potentials.t/ms,potentials.v.T[:,i*10]/mV)
    xlabel('Time (ms)')
    ylabel('v (mV)')
    title('Temporal evolution of membrane potential along a dendrite')
    figure()
    for i in range(t_sim):
        plot(my_neuron.distance/um,potentials.v.T[i*10,:]/mV)
    xlabel('X (um)')
    ylabel('v (mV)')
    title('Spatial distribution of membrance potential for different time instances')
    
    show()
    
    
#This function plots the stationary solution of a dendrite with a specific length L, with N compartments,
#stimulated from one end with a constant current of I0 nA amplitude.
#It also plots the analytical stationary solution for potential over the compartments.
def StationaryVoltage(N=200,I0=0.01,t_sim=100):
    
    # morphological and electrical parameters
    d=1*um #diameter of dendrite
    RL=100*ohm*cm #Intracellular medium resistance
    RT=30000*ohm*cm**2#cell mebrane resistance
    L=1000*um #length of dendrite
    EL=-70*mV #reversal potential
    #tau=30*ms
    C=1*uF/cm**2
    
    # a dendrite with diameter=d
    my_morphology=Cylinder(diameter=d,length=L,n=N)
    
    #Im is transmembrane current
    #Iext is  injected current at a specific position on dendrite
    my_equs='''
    Iext: amp (point current)
    Im=(EL-v)/RT : amp/meter**2
    '''
    my_neuron=SpatialNeuron(morphology=my_morphology,model=my_equs,Cm=C,Ri=RL)
    
    my_neuron.Iext[0]=I0*nA # Injecting a short pulse current in the middleof dendrite
    my_neuron.v=EL
    run(t_sim*ms)
    
    #Analytical stationary solution:
   
    lambdaa=my_neuron.space_constant[0]
    x = my_neuron.distance
    ra = 4*lambdaa*RL / (pi * d**2)
    analytical_solution =  EL+ra * my_neuron.Iext[0] * cosh((L - x) / lambdaa) / sinh(L / lambdaa)
    
    figure()
    plot(x/um, analytical_solution/mV, label='Analytical solution')
    legend()

    plot(my_neuron.distance/um,my_neuron.v/mV,label='Numerical solution')
    legend()
    xlabel('X (um)')
    ylabel('v (mV)')
    title('Stationary voltage over the dendrite')
    show()

    
    
    
    
  
#tests
SpatialAndTemporal_PotentialEvolution(N=200,I0=0.01)
StationaryVoltage(N=1000,I0=0.01,t_sim=200)

