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

from numpy import *
from pylab import *
from brian import *

def get_trajectory(v0=0.,w0=0.,I0=0.,epsilon=0.1,a=2.0):
    """
        solve the FitzHugh Nagumo model (FHN)
        
        Inputs:
        v0      - intial condition for v
        w0      - intial condition for w
        I0      - intensity of the constant input current
        epsilon - time constant of the recovery variable w for the FitzHugh Nagumo model
        a       - offset of the w-nullcline
        
        Outputs:
        v   - voltage (membrane potential)
        w   - recovery variable
        """
    
   
    L = 500
     
    # forming the neuron model using differential equations
    eqs = '''
    I_e : amp  
    dv/dt = (v*( 1 - (v**2)/(mV**2)) - w + I_e*Mohm)/(ms) : volt
    dw/dt = epsilon*(v + 0.5*(a*mV - w))/ms : volt
    '''
    
    neuron = NeuronGroup(1, eqs)
    
    # initialization of simulator
    reinit()
    
    # parameter initialization
    neuron.v = v0 * mV
    neuron.w = w0 * mV
    
    # injecting current to the neuron
    neuron.I_e = I0 * nA
    
    # tracking parameters
    tracev = StateMonitor(neuron, 'v', record=True)
    tracew = StateMonitor(neuron, 'w', record=True)
   
    # running the simulation
    run(L * ms)
    
    return (tracev[0]/mV,tracew[0]/mV)


def plot_flow(I0=0.,epsilon=0.1,a=2.0):
    """
        Plot FitzHugh Nagumo phase plane along with the trajectory of the voltage
        when the model is stimulated by a constant current I
        
        Inputs:
        I0      - amplitude of current step
        epsilon - time constant of the recovery variable w for the FitzHugh Nagumo model
        a       - offset of the w-nullcline
        """
    
    # define the interval spanned by voltage v and recovery variable w
    # to produce the phase plane
    vv = arange(-2.5,2.5,0.2)
    ww = arange(-2.5,5.5,0.2)
    (VV,WW) = np.meshgrid(vv,ww)
    
    # Compute derivative of v and w according to equation 1 and mean velocity vel
    dV = VV*( 1 - (VV**2)) - WW + I0
    dW = epsilon*(VV + 0.5*(a - WW))
    vel = sqrt(dV**2+dW**2)

    # Use quiver function to plot the phase plane
    quiver(VV,WW,dV,dW,vel)


def get_fixed_point(I0,epsilon=0.1,a=2.0):
    """
        Compute the fixed point of the FitzHugh Nagumo model as a function of the input current I0
        To do so, solve the 3rd order poylnomial equation:
        v**3 + V + a - I0 = 0
        Inputs:
        I0      - amplitude of current step
        epsilon - time constant of the recovery variable w for the FitzHugh Nagumo model
        a       - offset of the w-nullcline
        """
    
    # Use poly1d function from numpy to compute the roots of 3rd order polynomial
    P = poly1d([1, 0, 1, (a-I0)], variable = 'x')
    vStar = real(P.r[isreal(P.r)])[0]   # take only the real root
    wStar = 2.*vStar + a
    return (vStar,wStar)


string = "      ***********************************************************"
string += "\r\n"
string += "      ***** Ecole polytechnique federale de Lausanne (EPFL) *****"
string += "\r\n"
string += "      ***** Laboratory of computational neuroscience (LCN)  *****"
string += "\r\n"
string += "      *****               Neuronal Dynamics                 *****"
string += "\r\n"
string += "      ***********************************************************"
string += "\r\n"
string += "This file implements FitzHugh Nagumo model. It contains below methods:"
string += "\r\n"
string += "get_trajecotry()  : Calculate the  trajectory of parameters  using initial values"
string += "\r\n"
string += "plot_flow()       : Plot the phase plane along with the trajectory of the voltage"
string += "\r\n"
string += "get_fixed_point() : Compute the fixed point of the model "
string += "\r\n\r\n"
string += "In order to know parameters and default values for each method use symbol ? after"
string += "\r\n"
string += "the name of method. For example: plot_flow?"
string += "\r\n\r\n"
string += "----------------------2014 EPFL-LCN --------------------------"
string += "\r\n"

print string
