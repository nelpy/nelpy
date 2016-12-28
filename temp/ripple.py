# ripple.py
# Caleb Kemere

import numpy as np
import scipy.signal as signal
import scipy.ndimage
from itertools import groupby
from operator import itemgetter

def find_threshold_crossing_events(x, threshold) :
    above_threshold = np.where(x > threshold, 1, 0);
    eventlist = []
    eventmax = []
    for k,v in groupby(enumerate(above_threshold),key=itemgetter(1)):
        if k:
            v = list(v)
            eventlist.append([v[0][0],v[-1][0]])
            try :
                eventmax.append(x[v[0][0]:(v[-1][0]+1)].max())
            except :
                print(v, x[v[0][0]:v[-1][0]])
                
    eventmax = np.asarray(eventmax)
    eventlist = np.asarray(eventlist)
    return eventlist, eventmax


def detect(data, FS=1500, ThresholdSigma=3, SecondaryThresholdSigma=0, LengthCriteria=0.015):
    #  (1) filter 150-250
    #  (2) hilbert envelope
    #  (3) smooth with Gaussian (4 ms SD)
    #  (4) 3SD above the mean for 15 ms
    #  (5) full ripple defined as window back to mean

    # Generate filter for detection
    b = signal.firwin(25, [150/(FS/2), 250/(FS/2)], pass_zero=False)
    # Filter raw data to get ripple data
    ripple_data = signal.filtfilt(b,1,data)
    # Use hilbert transform to get an envelope
    ripple_envelope = np.absolute(signal.hilbert(ripple_data))

    # Smooth envelope with a gaussian
    EnvelopeSmoothingSD = 0.004 * FS
    smoothed_envelope = scipy.ndimage.filters.gaussian_filter1d(ripple_envelope, EnvelopeSmoothingSD, mode='constant')

    # Find periods where value is > mean + ThresholdSigma*SD
    Threshold = np.mean(smoothed_envelope) + ThresholdSigma*np.std(smoothed_envelope)
    ripple_events, _ = find_threshold_crossing_events(smoothed_envelope, Threshold)

    # Keep only events that are long enough (LengthCriteria s)
    ripple_events = \
        ripple_events[ripple_events[:,1] - ripple_events[:,0] >= np.round(FS*LengthCriteria),:]


    # Find periods where value is > SecondaryThreshold; note that the previous periods should be within these!
    assert SecondaryThresholdSigma < ThresholdSigma, "Secondary Threshold by definition should include more data than Primary Threshold"
    SecondaryThreshold = np.mean(smoothed_envelope) + SecondaryThresholdSigma * np.std(smoothed_envelope)
    ripple_bounds, broader_maxes = find_threshold_crossing_events(smoothed_envelope, SecondaryThreshold)

    # Find corresponding big windows for potential ripple events
    #  Specifically, look for closest left edge that is just smaller
    outer_boundary_indices = np.searchsorted(ripple_bounds[:,0], ripple_events[:,0]);
    #  searchsorted finds the index after, so subtract one to get index before
    outer_boundary_indices = outer_boundary_indices - 1;


    # Find extended boundaries for ripple events by pairing to larger windows
    #   (Note that there may be repeats if the larger window contains multiple > 3SD sections)
    ripple_bounds = ripple_bounds[outer_boundary_indices,:]
    ripple_maxes = broader_maxes[outer_boundary_indices]

    # Now, since all that we care about are the larger windows, so we should get rid of repeats
    _, unique_idx = np.unique(ripple_bounds[:,0], return_index=True)
    ripple_bounds = ripple_bounds[unique_idx,:]
    ripple_maxes = ripple_maxes[unique_idx]
    ripple_events = ripple_events[unique_idx,:]

    return ripple_bounds, ripple_maxes, ripple_events, ripple_data, ripple_envelope, smoothed_envelope





