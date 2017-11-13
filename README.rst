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

Support
=======
This work was supported by the National Science Foundation (CBET-1351692 and IOS-1550994) and the Human Frontiers Science Program (RGY0088). Any opinions, findings, and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the National Science Foundation.

Quick examples
==============

Let's give it a try. Create a ``SpikeTrainArray``:

.. code-block:: python

    import nelpy as nel  # main nelpy imports
    import nelpy.plotting as npl  # optional plotting imports
    spike_times = np.array([1, 2, 4, 5, 10])
    st = nel.SpikeTrainArray(spike_times)

Do something:

.. code-block::

    >>> print(st.n_spikes) # print out how many spikes there are in st
    5

    >>> print(st.supportn_spikes) # print out the underlying EpochArray on which st is defined
    <EpochArray at 0x1d4812c7550: 1 epoch> of duration 9 seconds

    >>> npl.raster(st) # plots the spike raster


As a more representative example of what nelpy can be used for, consider the estimation of
place fields (spatial tuning curves) of CA1 units while an animal runs on a linear track.

Estimating the place fields can be a complicated affair, and roughly involves the following steps:

1. assume we have position data and spike data available
2. linearize the environment (and position data), if desired
3. estimate the running velocity from the position data
4. smooth the velocity estimates, since numerical differentiation is inherently noisy (and our measurements are imprecise)
5. identify epochs where the animal was running, and where the animal was resting
6. count the number of spikes from each unit, in each spatial bin of the environment, during run behavior
7. determine how long the animal spent in each spatial bin (while running)
8. estimate a firing rate within each spatial bin, by normalizing the number of observed spikes by the time spent in that spatial bin
9. visualize the estimated tuning curves, and evaluate how well the tuning curves can be used to decode the animal's position
10. ...

.. class:: no-web

    .. image:: https://raw.githubusercontent.com/nelpy/nelpy/develop/.placefields.png
        :alt: nelpy-promo-pic
        :width: 100%
        :align: center

Nelpy makes it easy to do all of the above, to interact with the ephys data, and to visualize the results.

To see the full code that was used to generate the figures above, take a look at the `linear track example analysis <https://github.com/nelpy/example-analyses/blob/master/LinearTrackDemo.ipynb>`_.

Getting started
===============
The best way to get started with using ``nelpy`` is probably to take a look at
the `tutorials <https://github.com/nelpy/tutorials>`_ and
`example analyses <https://github.com/nelpy/example-analyses>`_.

The tutorials are still pretty bare-bones, but will hopefully be expanded soon!

Installation
============

The easiest way to install nelpy is to use ``pip``. From the terminal, run:

.. code-block:: bash

    $ pip install nelpy

Alternatively, you can install the latest version of nelpy by running the following commands:

.. code-block:: bash

    $ git clone https://github.com/nelpy/nelpy.git
    $ cd nelpy
    $ python setup.py [install, develop]

where the ``develop`` argument should be used if you want to modify the code.

A weak prerequisite for installing nelpy is a modified version of `hmmlearn <https://github.com/eackermann/hmmlearn/tree/master/hmmlearn>`_. This requirement is weak, in the sense that installation will complete successfully without it, and most of nelpy can also be used without any problems. However, as soon as any of the hidden Markov model (HMM) functions are used, you will get an error if the correct version of ``hmmlearn`` is not installed. To make things easier, there is a handy 64-bit Windows wheel in the `hmmlearn directory <https://github.com/nelpy/nelpy/blob/master/hmmlearn/>`_ of this repository. Installation on Linux/Unix should be almost trivial.

Related work and inspiration
============================
Nelpy drew heavy inspiration from the ``python-vdmlab`` package (renamed to ``nept``)
from the van der Meer lab at Dartmouth College (https://github.com/vandermeerlab),
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

Where
=====

===================   ========================================================
 download             https://pypi.python.org/pypi/nelpy
 tutorials            https://github.com/nelpy/tutorials
 example analyses     https://github.com/nelpy/example-analyses
 docs                 coming soon!
 code                 https://github.com/nelpy/nelpy
===================   ========================================================

License
=======

Nelpy is distributed under the MIT license. See the `LICENSE <https://github.com/nelpy/nelpy/blob/master/LICENSE>`_ file for details.
