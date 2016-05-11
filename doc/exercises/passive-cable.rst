Compartment Model for Passive Cable
===========================

The goal of these exercises is to acquire some familiarity with
`compartmental neuron
models <http://neuronaldynamics.epfl.ch/online/Ch3.S4.html>`__. Download
CompartmentalModel\_Ex.py from
`here <http://neuronaldynamics.epfl.ch/lectures.html>`__.
CompartmentalModel\_Ex.py is a python module containing 2 main
functions: SpatialAndTemporal\_PotentialEvolution and StationaryVoltage.
With those, you can plot spatial and temporal evolution of voltage and
stationary voltage, respectively. Once you have started ipython -pylab
in the directory containing CompartmentalModel\_Ex.py, simply type:

::

    >> import CompartmentalModel_Ex

| to port CompartmentalModel\_Ex.py onto your current session.
| Then call the functions simply by typing:

::

    >> CompartmentalModel_Ex.SpatialAndTemporal_PotentialEvolution ()

or

::

    >> CompartmentaModel_Ex.StationaryVoltage()

| The aim of this exercise is to numerically solve the general cable
  equation. Refer to equation 3.13 for cable equation from
  `here <http://neuronaldynamics.epfl.ch/online/Ch3.S2.html>`__.
| To do so, we can define a neuron model with extended morphology (with
  different compartments) using *SpatialNeuron* class of *Brian2*
  simulator. *SpatialNeuron* is defined mainly by a set of equations
  describing transmembrane (and possibly other) currents and a
  morphology.
| Use the function *SpatialAndTemporal\_PotentialEvolution(N,I0,t\_sim)*
  which defines a cable with specific length and N compartments which
  will be stimulated with a short pulse of amplitude I0 from its middle.
  With calling that function you can observe temporal and spatial
  evolution of potential. Describe how the potential changes over
  different compartments, as the time goes on. I.e. can you discriminate
  which curve in spatial distribution plot, is related to the earliest
  time instance, and which curve in time evolution plot corresponds to a
  further position with respect to the stimulation point by current?

| The aim of this exercise is to find the numerical stationary solution
  of voltage over compartments of a cable, and compare it to its
  analytical solution. To do this, call the function
  *StationaryVoltage(N,I0,t\_sim)*, which creates a cable with specific
  length, e.g. a dendrite, with N compartments and stimulate it from one
  end with constant current I0.
| Try different values for N and observe how the numerical solution
  changes with respect to the analytical solution? How can you justify
  your observation? (For example you can try N=[10,100,1000]. ) Increase
  amplitude of injected current, what do you observe?
