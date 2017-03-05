=====
Nelpy
=====

Neuroelectrophysiology object model, data exploration, and analysis in Python.

Overview
========
Nelpy (**N**\ euro\ **el**\ ectro\ **p**\ hysiolog\ **y**) is an open source package for analysis of neuroelectrophysiology data. Nelpy defines a number of data objects to make it easier to work with electrophysiology (ephys) data, and although it was originally designed for use with extracellular recorded data, it can be used much more broadly. Nelpy is intended to make interactive data analysis and exploration of these ephys data easy, by providing several convenience functions and common visualizations that operate directly on the nelpy objects.

More specifically, the functionality of this package includes:

- several container objects (``SpikeTrain``, ``BinnedSpikeTrain``, ``AnalogSignal``, ``EpochArray``, ...) with nice human-readable ``__repr__`` methods
- powerful ways to interact with the data in the container objects
- hidden Markov model analysis of neural activity
- basic data exploration and visualization operating directly on the core nelpy objects
- and much more

Quick example
=============

Let's give it a try. Create a ``SpikeTrainArray``:

.. code-block:: python

    import nelpy as nel  # main nelpy imports
    import nelpy.plotting as npl  # optional plotting imports
    spike_times = np.array([1, 2, 4, 5, 10])
    st = nel.SpikeTrainArray(spike_times, fs=1)

Do something:

.. code-block:: python

    st.n_spikes
    5

Related work and inspiration
============================
Nelpy drew heavy inspiration from the ``python-vdmlab`` package from the
van der Meer lab at Dartmouth College (https://github.com/vandermeerlab),
which was created by Emily Irvine (https://github.com/emirvine). It is
also inspired by the neuralensemble.org NEO project (http://neo.readthedocs.io).

**Short history:** Etienne A started the nelpy project for two main reasons, namely

1. he wanted / needed a ``BinnedSpikeTrain`` object for hidden Markov model analysis that wasn't (at the time) avaialable in ``neo`` or ``python-vdmlab``, and
2. he fundamentally wanted to add "support" attributes to all the container objects. Here "support" should be understood in the mathematical sense of "domain of definition", whereas the mathematical support technically would not include some elements for which the function maps to zero. This is critical for spike trains, for example, where it is important to differentiate "no spike at time t" from "no record at time t".

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

The easiest way to install nelpy is to use ``pip``. From the terminal, run:

.. code-block:: bash

    pip install nelpy

Alternatively, you can install the latest version of nelpy by running the following commands:

.. code-block:: bash

    git clone https://github.com/eackermann/nelpy.git
    cd nelpy
    python setup.py [install, develop]

where the ``develop`` argument should be used if you want to modify the code.

A weak prerequisite for installing nelpy is a modified version of ``hmmlearn``. This requirement is weak, in the sense that installation will complete successfully without it, and most of nelpy can also be used without any problems. However, as soon as any of the hidden Markov model (HMM) functions are used, you will get an error if the correct version of ``hmmlearn`` is not installed. To make things easier, there is a handy 64-bit Windows wheel in the hmmlearn directory of this repository. Installation on Linux/Unix should be almost trivial.

Where
=====

===================   ========================================================
 download             https://pypi.python.org/pypi/nelpy
 docs                 coming soon!
 code                 https://github.com/eackermann/nelpy
===================   ========================================================

License
=======

Nelpy is distributed under the MIT license. See the `LICENSE <https://github.com/eackermann/nelpy/blob/master/LICENSE>`_ file for details.
