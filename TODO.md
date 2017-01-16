TODO.md
===========================

1. add proper attribution (Emily Irvine) and figure out how to collaborate with her / vdmlab
1. implement AnalogSignal, LocalFieldPotential (with support) --- even AnalogSignal should have support though, so inherit from there
1. implement SpikeTrainArray (shared support)
1. implement LocalFieldPotentialArray (shared support)
1. demo LFP[SWRs] and LFP[noSWRs] ==> count cells active in SWR events?
1. implement Position (location? occupancy? 3D? velocity? subsample? linear interp? smooth?)
1. implement BinnedData class?
1. add tests using pytest
1. figure out how to profile speed and memory, and test
    1. lazy time calculation
    1. speed for realistic and edge cases of spike trains
1. re-factor EpochArray and SpikeTrain to handle empty objects more efficiently
1. track down merging bug
1. reconsider and / or deprecate SpikeTrain.time_slice and SpikeTrain.time_slices
1. look at svg_utils for grid util
1. plan how to use hierarchical indexing, and eval and query from Pandas for increased speed
1. consider forcing named arguments using ,* ,
1. fix documentation ito parameters and attributes
1. pepare and upload to github, along with setup.py and env details
1. example with train-test-epoch-split-and-merge, and integration with hmmlearn
1. change default parameters to =None
1. reconsider SpikeTrain.shift() --- could just as well create new SpikeTrain with explicitly shifted samples?
1. Should we follow NEO more closely to allow for Spike and Epoch objects? and what about Unit objects?
1. Update Emily's wonderful tutorial to showcase EpochArray indexing, class __repr__ output, SpikeTrain support, and more...

follow along here: https://jeffknupp.com/blog/2013/08/16/open-sourcing-a-python-project-the-right-way/

and here: http://nvie.com/posts/a-successful-git-branching-model/

http://neo.readthedocs.io/en/0.4.0/core.html
