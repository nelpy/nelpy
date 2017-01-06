Neuroelectrophysiology object model and analysis in Python
==========================================================

* Nelpy: **N**euro**EL**ectro**P**hysiolog**Y** in **PY**thon.
* Nelpy: Rice University **N**eural **E**ngineering **L**ab **PY**thon 
    object model and data anlysis.

First things first
==================

This project is based on the `python-vdmlab` package from the 
van der Meer lab at Dartmouth College (https://github.com/vandermeerlab),
created by Emily Irvine (https://github.com/emirvine). It is also inspired
by the neuralensemble.org NEO project (http://neo.readthedocs.io/en/0.4.0/).

Scope of this work
==================
The nelpy object model is expected to be quite similar to python-vdmlab object
model, which in turn has significant overlap with neuralensemble.org's neo 
model. However, the nelpy object model extends the former by making binned data
first class citizens, and by changing the API for indexing and extracting subsets
of data, as well as making "functional support" an integral part of the model. It
(nelpy) is impler and less well developed than neo, and specifically lacks in 
terms of physical units and complex object hierarchies and nonlinear relationships.
However, nelpy again makes binned data a core object, and nelpy further aims to 
add additional analysis code including filtering, smoothing, position analysis,
subsampling, interpolation, spike rate estimation, spike generation / synthesis,
ripple detection, Bayesian decoding, and so on. In short, nelpy is more than just
an object model, but the nelpy core is designed to be a flexible, readable, yet
powerful object model for neuroelectrophysiology.

Getting started
===============

* Instructions will go here...

Documentation
=============

Users
-----

Developers
----------

Testing
=======

License
=======

Nelpy is made available under the [MIT license](LICENSE) 
that allows using, copying, and sharing.

Projects using nelpy
====================

* [none](url)
