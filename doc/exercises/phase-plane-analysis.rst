FitzHugh-Nagumo: Phase plane and bifurcation analysis
=====================================================

**Book chapters**

See `Chapter 4 <Chapter4_>`_ and especially `Chapter 4 Section 3 <Chapter43_>`_ for background knowledge on phase plane analysis.

.. _Chapter4: http://neuronaldynamics.epfl.ch/online/Ch4.html
.. _Chapter43: http://neuronaldynamics.epfl.ch/online/Ch4.S3.html


**Python classes**

In this exercise we study the phase plane of a two dimensional dynamical system implemented in the module :mod:`.phase_plane_analysis.fitzhugh_nagumo`. To get started, copy the following code block into your Jupyter Notebook. Check the documentation to learn how to use these functions. Make sure you understand the parameters the functions take.

.. code-block:: python

      %matplotlib inline
      import brian2 as b2
      import matplotlib.pyplot as plt
      import numpy as np
      from neurodynex.phase_plane_analysis import fitzhugh_nagumo

      fitzhugh_nagumo.plot_flow()

      fixed_point = fitzhugh_nagumo.get_fixed_point()
      print("fixed_point: {}".format(fixed_point))

      plt.figure()
      trajectory = fitzhugh_nagumo.get_trajectory()
      plt.plot(trajectory[0], trajectory[1])



Exercise: Phase plane analysis
------------------------------
We have implemented the following Fitzhugh-Nagumo model.

.. math::
   :label: eq1

   \left[\begin{array}{ccll}
   {\displaystyle \frac{du}{dt}} &=& u\left(1-u^{2}\right)-w+I \equiv F(u,w)\\[.2cm]
   {\displaystyle \frac{dw}{dt}} &=& \varepsilon \left(u -0.5w+1\right) \equiv \varepsilon G(u,w)\, ,\\
   \end{array}\right.


Question
~~~~~~~~

Use the function :func:`plt.plot <matplotlib.pyplot.plot>` to plot the two nullclines of the Fitzhugh-Nagumo system given in Eq. :eq:`eq1` for :math:`I = 0` and :math:`\varepsilon=0.1`.

Plot the nullclines in the :math:`u-w` plane, for voltages in the region :math:`u\in\left[-2.5,2.5\right]`.

.. note::

	For instance the following example shows plotting the function
	:math:`y(x) = -\frac{x^2}{2} + x + 1`:

	.. code-block:: python
		
		x = np.arange(-2.5, 2.51, .1)  # create an array of x values
		y = -x**2 / 2. + x + 1  # calculate the function values for the given x values
		plt.plot(x, y, color='black')  # plot y as a function of x
		plt.xlim(-2.5, 2.5)  # constrain the x limits of the plot

	You can use similar code to plot the nullclines, inserting the appropriate equations.


.. _q-traj:

Question
~~~~~~~~

Get the lists ``t``, ``u`` and  ``w`` by calling :func:`t, u, w = fitzhugh_nagumo.get_trajectory(u_0, w_0, I) <.phase_plane_analysis.fitzhugh_nagumo.get_trajectory>` for :math:`u_0 = 0`, :math:`w_0= 0` and :math:`I = 1.3`. They are corresponding values of :math:`t`, :math:`u(t)` and :math:`w(t)` during trajectories starting at the given point :math:`(u_0,w_0)` for a given **constant** input current :math:`I`. Plot the nullclines for this given current and the trajectories into the :math:`u-w` plane.

Question
~~~~~~~~

At this point for the same current :math:`I`, call the function :func:`plot_flow <.phase_plane_analysis.fitzhugh_nagumo.plot_flow>`, which adds the flow created by the system Eq. :eq:`eq1` to your plot. This indicates the direction that trajectories will take.

.. note::

	If everything went right so far, the trajectories should follow the flow. First, create a new figure by calling :func:`plt.figure() <matplotlib.pyplot.plot>` and then plot the :math:`u` data points from the trajectory obtained in :ref:`the previous exercise <q-traj>` on the ordinate.

	You can do this by using the :func:`plt.plot <matplotlib.pyplot.plot>` function and passing only the array of :math:`u` data points:


	.. code-block:: python

		u = [1,2,3,4]  # example data points of the u trajectory
		plot(u, color='blue')  # plot will assume that u is the ordinate data

.. _q-traj2:

Question
~~~~~~~~

Finally, change the input current in your python file to other values :math:`I>0` and reload it. You might have to first define :math:`I` as a variable and then use this variable in all following commands if you did not do so already. At which value of :math:`I` do you observe the change in stability of the system?


Exercise: Jacobian & Eigenvalues
--------------------------------

The linear stability of a system of differential equations can be evaluated by calculating the eigenvalues of the system’s Jacobian at the fixed points. In the following we will graphically explore the linear stability of the fixed point of the system Eq. :eq:`eq1`. We will find that the linear stability changes as the input current crosses a critical value.

.. _q-jac:

Question
~~~~~~~~
Set :math:`\varepsilon=.1` and :math:`I` to zero for the moment. Then, the Jacobian of Eq. :eq:`eq1` as a function of the fixed point is
given by

.. math::

   \begin{aligned}
   J\left(u_{0},w_{0}\right) & = & \left.\left(\begin{array}{cc}
   1-3u_0^2 & -1\\[5pt]
   0.1 & -0.05
   \end{array}\right)\right.\end{aligned}

Write a python function ``get_jacobian(u_0,w_0)`` that returns
the Jacobian evaluated for a given fixed point :math:`(u_0,v_0)` as a
python list. 

.. note::
	An example for a function that returns a list
	corresponding to the matrix :math:`M(a,b)=\left(\begin{array}{cc}
	a & 1\\
	0 & b
	\end{array}\right)` is:

	.. code-block:: python

		def get_M(a,b):
			return [[a,1],[0,b]] # return the matrix


.. _q-jac2:

Question
~~~~~~~~

The function :func:`u0,w0 = get_fixed_point(I) <.phase_plane_analysis.fitzhugh_nagumo.get_fixed_point>` gives you the numerical coordinates of the fixed point for a given current :math:`I`. Use the function you created in :ref:`the previous exercise <q-jac>` to evaluate the Jacobian at this fixed point and store it in a new variable ``J``.

.. _q-jac3:

Question
~~~~~~~~

Calculate the eigenvalues of the Jacobian ``J``, which you computed in
:ref:`the previous exercise <q-jac2>`, by using the function :func:`np.linalg.eigvals(J) <numpy.linalg.eigvals>`. Both should be negative for :math:`I=0`.


Exercise: Bifurcation analysis
------------------------------

Wrap the code you wrote so far by a loop, to calculate the eigenvalues for increasing values of :math:`I`. Store the changing values of each eigenvalue in seperate lists, and finally plot their real values against :math:`I`. 

.. note::

	You can start from this example loop:
    .. code-block:: py

        import numpy as np
        list1 = []
        list2 = []
        currents = np.arange(0,4,.1) # the I values to use
        for I in currents:
            # your code to calculate the eigenvalues e = [e1,e2] for a given I goes here
            list1.append(e[0].real) # store each value in a separate list
            list2.append(e[1].real)

        # your code to plot list1 and list 2 against I goes here


Question
~~~~~~~~

In what range of :math:`I` are the real parts of eigenvalues positive?

Question
~~~~~~~~

Compare this :ref:`with your earlier result <q-traj2>` for the critical :math:`I`. What does this imply for the stability of the fixed point? What has become stable in this system instead of the fixed point?