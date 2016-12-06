"""
This file implements functions to simulate and analyze
Fitzhugh-Nagumo type differential equations with Brian2.

Relevant book chapters:

- http://neuronaldynamics.epfl.ch/online/Ch4.html
- http://neuronaldynamics.epfl.ch/online/Ch4.S3.html.

"""

# This file is part of the exercise code repository accompanying
# the book: Neuronal Dynamics (see http://neuronaldynamics.epfl.ch)
# located at http://github.com/EPFL-LCN/neuronaldynamics-exercises.

# This free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License 2.0 as published by the
# Free Software Foundation. You should have received a copy of the
# GNU General Public License along with the repository. If not,
# see http://www.gnu.org/licenses/.

# Should you reuse and publish the code for your own purposes,
# please cite the book or point to the webpage http://neuronaldynamics.epfl.ch.

# Wulfram Gerstner, Werner M. Kistler, Richard Naud, and Liam Paninski.
# Neuronal Dynamics: From Single Neurons to Networks and Models of Cognition.
# Cambridge University Press, 2014.

import brian2 as b2
import matplotlib.pyplot as plt
import numpy as np


def get_trajectory(v0=0., w0=0., I=0., eps=0.1, a=2.0, tend=500.):
    """Solves the following system of FitzHugh Nagumo equations
    for given initial conditions:

    dv/dt = 1/1ms * v * (1-v**2) - w + I
    dw/dt = eps * (v + 0.5 * (a - w))

    Args:
        v0: Intial condition for v [mV]
        w0: Intial condition for w [mV]
        I: Constant input [mV]
        eps: Inverse time constant of the recovery variable w [1/ms]
        a: Offset of the w-nullcline [mV]
        tend: Simulation time [ms]

    Returns:
        tuple: (t, v, w) tuple for solutions
    """

    eqs = """
    I_e : amp
    dv/dt = 1/ms * ( v * (1 - (v**2) / (mV**2) ) - w + I_e * Mohm ) : volt
    dw/dt = eps/ms * (v + 0.5 * (a * mV - w)) : volt
    """

    neuron = b2.NeuronGroup(1, eqs, method="euler")

    # state initialization
    neuron.v = v0 * b2.mV
    neuron.w = w0 * b2.mV

    # set input current
    neuron.I_e = I * b2.nA

    # record states
    rec = b2.StateMonitor(neuron, ["v", "w"], record=True)

    # run the simulation
    b2.run(tend * b2.ms)

    return (rec.t / b2.ms, rec.v[0] / b2.mV, rec.w[0] / b2.mV)


def plot_flow(I=0., eps=0.1, a=2.0):
    """Plots the phase plane of the Fitzhugh-Nagumo model
    for given model parameters.

    Args:
        I: Constant input [mV]
        eps: Inverse time constant of the recovery variable w [1/ms]
        a: Offset of the w-nullcline [mV]
    """

    # define the interval spanned by voltage v and recovery variable w
    # to produce the phase plane
    vv = np.arange(-2.5, 2.5, 0.2)
    ww = np.arange(-2.5, 5.5, 0.2)
    (VV, WW) = np.meshgrid(vv, ww)

    # Compute derivative of v and w according to FHN equations
    # and velocity as vector norm
    dV = VV * (1. - (VV**2)) - WW + I
    dW = eps * (VV + 0.5 * (a - WW))
    vel = np.sqrt(dV**2 + dW**2)

    # Use quiver function to plot the phase plane
    plt.quiver(VV, WW, dV, dW, vel)


def get_fixed_point(I=0., eps=0.1, a=2.0):
    """Computes the fixed point of the FitzHugh Nagumo model
    as a function of the input current I.

    We solve the 3rd order poylnomial equation:
    v**3 + V + a - I0 = 0

    Args:
        I: Constant input [mV]
        eps: Inverse time constant of the recovery variable w [1/ms]
        a: Offset of the w-nullcline [mV]

    Returns:
        tuple: (v_fp, w_fp) fixed point of the equations
    """

    # Use poly1d function from numpy to compute the
    # roots of 3rd order polynomial
    P = np.poly1d([1, 0, 1, (a - I)], variable="x")

    # take only the real root
    v_fp = np.real(P.r[np.isreal(P.r)])[0]
    w_fp = 2. * v_fp + a

    return (v_fp, w_fp)
