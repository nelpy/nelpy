Neuroelectrophysiology object model and analysis in Python
==========================================================

* Nelpy: Neuroelectrophysiology in Python.

Nelpy defines a few fundamental data objects to make it easier to work
with electrophysiology (ephys) data. It was originally designed for use
with extracellular recorded data using implanted electrodes, but it can
be much more broadly applicable.

In addition, nelpy is intended to make **interactive** data analysis and
exploration of these ephys data easy, by providing several convenience
functions and common visualization functions that operate directly on
the nelpy objects.

First things first
==================

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

Getting started
===============

* The easiest way is to use `pip install nelpy`, but alternatively you can also do

```
git clone https://github.com/eackermann/nelpy.git
```

followed by

```
cd nelpy
python setup.py install
```

and then, following successful installation, simply import nelpy:

```
>>> import nelpy as nel  # main nelpy imports
>>> import nelpy.plotting as npl  # optional plotting imports
```

That's it!

Documentation
=============
Coming soon! But the code is already pretty well documented.

Users
-----

Developers
----------
See [here](developnotes.md)

Testing
-------
Coming soon! Expected in release 0.1.0