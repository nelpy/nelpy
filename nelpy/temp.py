def drop_last_bin(bst, inplace=False):
    """Drop the last bin from a BinnedSpikeTrainArray with a single epoch."""
    if bst.isempty:
        return bst
    assert bst.n_epochs == 1, 'Not supported on BSTs with multiple epochs!'

    if inplace:
        out = bst
    else:
        from copy import deepcopy
        out = deepcopy(bst)

    out._support = bst._support.shrink(bst.ds, direction='stop') # shrink support(s) by one bin size, on right
    out._bin_centers = bst._bin_centers[:-1] # remove last bin center NB! this operates on last epoch only!
    out._binnedSupport[:,1] = bst._binnedSupport[:,1] - 1 # remove last bin from each epoch
    out._bins = bst._bins[:-1]
    out._data = bst._data[:,:-1]

    return out

def detect_ripples(eeg):
    """Detect sharp wave ripples (SWRs) from single channel eeg (AnalogSignalArray).
    """

    # Maggie defines ripples by doing:
    #  (1) filter 150-250
    #  (2) hilbert envelope
    #  (3) smooth with Gaussian (4 ms SD)
    #  (4) 3.5 SD above the mean for 15 ms
    #  (5) full ripple defined as window back to mean

    assert eeg.n_signals == 1, "only single channel ripple detection currently supported!"

    # (1)
    ripple_eeg = nel.filtering.sosfiltfilt(eeg, fl=150, fh=250)
    # (2, 3)
    ripple_envelope = nel.utils.signal_envelope1D(ripple_eeg, sigma=0.004)
    # (4, 5)
    bounds, maxes, events = nel.utils.get_events_boundaries(
                x=ripple_envelope.ydata,
                PrimaryThreshold=ripple_envelope.mean() + 3.5*ripple_envelope.std(),   # cm/s
                SecondaryThreshold=ripple_envelope.mean(),  # cm/s
                minThresholdLength=0.015, # threshold crossing must be at least 15 ms long
                minLength=0.0, # total ripple duration must be at least XXX ms long
                ds = 1/ripple_envelope.fs
            )

    # convert bounds to time in seconds
    timebounds = ripple_envelope.time[bounds]

    # add 1/fs to stops for open interval
    timebounds[:,1] += 1/eeg.fs

    # create EpochArray with bounds
    ripple_epochs = nel.EpochArray(timebounds)
    return ripple_epochs
