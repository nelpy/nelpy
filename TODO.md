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
1. Update Emily's wonderful tutorial to showcase EpochArray indexing, class repr output, SpikeTrain support, and more...
1. implement gettatr steps for slices (see below); also, what is the tuple usage?
1. see https://github.com/kghose/neurapy/blob/master/neurapy/signal/continuous.py for long filtfilt

follow along here: https://jeffknupp.com/blog/2013/08/16/open-sourcing-a-python-project-the-right-way/

and here: http://nvie.com/posts/a-successful-git-branching-model/

http://neo.readthedocs.io/en/0.4.0/core.html

def __getitem__(self, i):
        '''
        Get the item or slice :attr:`i`.
        '''
        obj = super(AnalogSignal, self).__getitem__(i)
        if isinstance(i, int):  # a single point in time across all channels
            obj = pq.Quantity(obj.magnitude, units=obj.units)
        elif isinstance(i, tuple):
            j, k = i
            if isinstance(j, int):  # extract a quantity array
                obj = pq.Quantity(obj.magnitude, units=obj.units)
            else:
                if isinstance(j, slice):
                    if j.start:
                        obj.t_start = (self.t_start +
                                       j.start * self.sampling_period)
                    if j.step:
                        obj.sampling_period *= j.step
                elif isinstance(j, np.ndarray):
                    raise NotImplementedError("Arrays not yet supported")
                    # in the general case, would need to return IrregularlySampledSignal(Array)
                else:
                    raise TypeError("%s not supported" % type(j))
                if isinstance(k, int):
                    obj = obj.reshape(-1, 1)
        elif isinstance(i, slice):
            if i.start:
                obj.t_start = self.t_start + i.start * self.sampling_period
        else:
            raise IndexError("index should be an integer, tuple or slice")
        return obj
