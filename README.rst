=====
Nelpy
=====

Neuroelectrophysiology object model and analysis in Python.

Overview
========
Nelpy (NeuroELectroPhysiologY) is an open source package for analysis of
neuroelectrophysiology data arising (primarily) from extracellular
electrode recordings during neuroscience experiments. The functionality
of this package includes:

- item 1
- item 2
- hidden Markov model analysis of neural activity
- basic data exploration and visualization operating directly on the core nelpy objects

Quick example
=============

Let's give it a try. Create a SpikeTrainArray::

    >>> import nelpy as nel  # main nelpy imports
    >>> import nelpy.plotting as npl  # optional plotting imports
    >>> spike_times = np.array([1, 2, np.nan, 4, 5])

Do something::

    >>> st.n_units
    3.0

Related work and inspiration
============================
Nelpy drew heavy inspiration from the `python-vdmlab` package from the
van der Meer lab at Dartmouth College (https://github.com/vandermeerlab),
which was created by Emily Irvine (https://github.com/emirvine). It is
also inspired by the neuralensemble.org NEO project (http://neo.readthedocs.io).

Scope of this work
==================
The nelpy object model is expected to be quite similar to the python-vdmlab object
model, which in turn has significant overlap with neuralensemble.org's neo
model. However, the nelpy object model extends the former by making binned data
first class citizens, and by changing the API for indexing and extracting subsets
of data, as well as making "functional support" an integral part of the model. It
(nelpy) is currently simpler and less comprehensive than neo, and specifically lacks in
terms of physical units and complex object hierarchies and nonlinear relationships.
However, nelpy again makes binned data a core object, and nelpy further aims to
add additional analysis code including filtering, smoothing, position analysis,
subsampling, interpolation, spike rate estimation, spike generation / synthesis,
ripple detection, Bayesian decoding, and so on. In short, nelpy is more than just
an object model, but the nelpy core is designed to be a flexible, readable, yet
powerful object model for neuroelectrophysiology.

Installation
============

The easiest way to install Nelpy is to use pip. From the terminal, run::

    pip install nelpy

Alternatively, you can install the latest version of nelpy by running the following commands::

    git clone https://github.com/eackermann/nelpy.git
    cd nelpy
    python setup.py [install, develop]

where the `develop` argument should be used if you want to modify the code.

Where
=====

===================   ========================================================
 download             https://pypi.python.org/pypi/nelpy
 docs                 coming soon!
 code                 https://github.com/eackermann/nelpy
===================   ========================================================

License
=======

Nelpy is distributed under the MIT license. See the `LICENSE <LICENSE>`_ file for details.
