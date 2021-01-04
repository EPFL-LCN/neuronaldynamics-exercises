"""
Implements compartmental model of a passive cable. See Neuronal Dynamics
`Chapter 3 Section 2 <http://neuronaldynamics.epfl.ch/online/Ch3.S2.html>`_

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
from neurodynex3.tools import input_factory
import matplotlib.pyplot as plt
import numpy as np

# integration time step in milliseconds
b2.defaultclock.dt = 0.01 * b2.ms

# DEFAULT morphological and electrical parameters
CABLE_LENGTH = 500. * b2.um  # length of dendrite
CABLE_DIAMETER = 2. * b2.um  # diameter of dendrite
R_LONGITUDINAL = 0.5 * b2.kohm * b2.mm  # Intracellular medium resistance
R_TRANSVERSAL = 1.25 * b2.Mohm * b2.mm ** 2  # cell membrane resistance (->leak current)
E_LEAK = -70. * b2.mV  # reversal potential of the leak current (-> resting potential)
CAPACITANCE = 0.8 * b2.uF / b2.cm ** 2  # membrane capacitance
DEFAULT_INPUT_CURRENT = input_factory.get_step_current(2000, 3000, unit_time=b2.us, amplitude=0.2 * b2.namp)
DEFAULT_INPUT_LOCATION = [CABLE_LENGTH / 3]  # provide an array of locations
# print("Membrane Timescale = {}".format(R_TRANSVERSAL*CAPACITANCE))


def simulate_passive_cable(current_injection_location=DEFAULT_INPUT_LOCATION, input_current=DEFAULT_INPUT_CURRENT,
                           length=CABLE_LENGTH, diameter=CABLE_DIAMETER,
                           r_longitudinal=R_LONGITUDINAL,
                           r_transversal=R_TRANSVERSAL, e_leak=E_LEAK, initial_voltage=E_LEAK,
                           capacitance=CAPACITANCE, nr_compartments=200, simulation_time=5 * b2.ms):
    """Builds a multicompartment cable and numerically approximates the cable equation.

    Args:
        t_spikes (int): list of spike times
        current_injection_location (list): List [] of input locations (Quantity, Length): [123.*b2.um]
        input_current (TimedArray): TimedArray of current amplitudes. One column per current_injection_location.
        length (Quantity): Length of the cable: 0.8*b2.mm
        diameter (Quantity): Diameter of the cable: 0.2*b2.um
        r_longitudinal (Quantity): The longitudinal (axial) resistance of the cable: 0.5*b2.kohm*b2.mm
        r_transversal (Quantity): The transversal resistance (=membrane resistance): 1.25*b2.Mohm*b2.mm**2
        e_leak (Quantity): The reversal potential of the leak current (=resting potential): -70.*b2.mV
        initial_voltage (Quantity): Value of the potential at t=0: -70.*b2.mV
        capacitance (Quantity): Membrane capacitance: 0.8*b2.uF/b2.cm**2
        nr_compartments (int): Number of compartments. Spatial discretization: 200
        simulation_time (Quantity): Time for which the dynamics are simulated: 5*b2.ms

    Returns:
        (StateMonitor, SpatialNeuron): The state monitor contains the membrane voltage in a
        Time x Location matrix. The SpatialNeuron object specifies the simulated neuron model
        and gives access to the morphology. You may want to use those objects for
        spatial indexing: myVoltageStateMonitor[mySpatialNeuron.morphology[0.123*b2.um]].v
    """
    assert isinstance(input_current, b2.TimedArray), "input_current is not of type TimedArray"
    assert input_current.values.shape[1] == len(current_injection_location),\
        "number of injection_locations does not match nr of input currents"

    cable_morphology = b2.Cylinder(diameter=diameter, length=length, n=nr_compartments)
    # Im is transmembrane current
    # Iext is  injected current at a specific position on dendrite
    EL = e_leak
    RT = r_transversal
    eqs = """
    Iext = current(t, location_index): amp (point current)
    location_index : integer (constant)
    Im = (EL-v)/RT : amp/meter**2
    """
    cable_model = b2.SpatialNeuron(morphology=cable_morphology, model=eqs, Cm=capacitance, Ri=r_longitudinal)
    monitor_v = b2.StateMonitor(cable_model, "v", record=True)

    # inject all input currents at the specified location:
    nr_input_locations = len(current_injection_location)
    input_current_0 = np.insert(input_current.values, 0, 0., axis=1) * b2.amp  # insert default current: 0. [amp]
    current = b2.TimedArray(input_current_0, dt=input_current.dt * b2.second)
    for current_index in range(nr_input_locations):
        insert_location = current_injection_location[current_index]
        compartment_index = int(np.floor(insert_location / (length / nr_compartments)))
        # next line: current_index+1 because 0 is the default current 0Amp
        cable_model.location_index[compartment_index] = current_index + 1

    # set initial values and run for 1 ms
    cable_model.v = initial_voltage
    b2.run(simulation_time)
    return monitor_v, cable_model


def getting_started():
    """A simple code example to get started.
    """
    current = input_factory.get_step_current(500, 510, unit_time=b2.us, amplitude=3. * b2.namp)
    voltage_monitor, cable_model = simulate_passive_cable(
        length=0.5 * b2.mm, current_injection_location=[0.1 * b2.mm], input_current=current,
        nr_compartments=100, simulation_time=2 * b2.ms)

    # provide a minimal plot
    plt.figure()
    plt.imshow(voltage_monitor.v / b2.volt)
    plt.colorbar(label="voltage")
    plt.xlabel("time index")
    plt.ylabel("location index")
    plt.title("vm at (t,x), raw data voltage_monitor.v")
    plt.show()


if __name__ == "__main__":
    getting_started()
