Oja's hebbian learning rule
===========================

**Book chapters**

See `Chapter 19 Section 2 <Chapter_>`_ on the learning rule of Oja.

.. _Chapter: http://neuronaldynamics.epfl.ch/online/Ch19.S2.html#SS1.p6


**Python classes**

The :mod:`.ojas_rule.oja` module contains all code required for this exercise.
At the beginning of your exercise solution file, import the contained functions by

.. code-block:: py

	from neurodynex.ojas_rule.oja import *

You can then simply run the exercise functions by executing, e.g.

.. code-block:: py

	cloud = make_cloud()  # generate data points
	wcourse = learn(cloud)  # learn weights and return timecourse

Exercise: Circular data
-----------------------

Use the functions :func:`make_cloud <.ojas_rule.oja.make_cloud>` and :func:`learn <.ojas_rule.oja.learn>` to get the timecourse for weights that are learned on a **circular** data cloud (``ratio=1``). Plot the time course
of both components of the weight vector. Repeat this many times (:func:`learn <.ojas_rule.oja.learn>` will choose random initial conditions on each run), and plot this into the same plot. Can you explain what happens?


Exercise: Elliptic data
-----------------------


Repeat the previous question with an **elongated** elliptic data cloud (e.g. ``ratio=0.3``). Again, repeat this several times. 

Question
~~~~~~~~

What difference in terms of learning do you observe with respect to the circular data clouds?

Question
~~~~~~~~

Try to change the orientation of the ellipsoid (try several different angles). Can you explain what Oja's rule does?

.. note::
	To gain more insight, plot the learned weight vector in 2D space, and relate its orientation to that of the ellipsoid of data clouds.

Exercise: Non-centered data
---------------------------

The above exercises assume that the input activities can be negative (indeed the inputs were always statistically centered). In actual neurons, if we think of their activity as their firing rate, this cannot be less than zero.

Try again the previous exercise, but applying the learning rule on a noncentered data cloud. E.g., use ``5 + make_cloud(...)``, which centers the data around ``(5,5)``. What conclusions can you draw? Can you think of a modification to the learning rule?