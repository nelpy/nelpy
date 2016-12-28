def saveData(data, filename, verbose=True, overwrite=False):
    # TODO: check if file exists, and warns user, also give meaningful errors
    
    import pickle
    import os.path

    my_file = 'data/' + filename + '.pkl'
    
#     if os.path.isfile(myfile):
    
#     if my_file.is_file():
    if os.path.isfile(my_file):
        # file exists
        print('file already exists!')
        
        if overwrite:
            with open(my_file, 'wb') as handle:
                pickle.dump(data, file=handle)
            
            if verbose:
                print('data saved successfully... [using overwrite]')
    else:
        with open(my_file, 'wb') as handle:
            pickle.dump(data, file=handle)
        
        if verbose:
            print('data saved successfully...')



##################################
#   new data loading interface   #
##################################

def get_exp_kws_from_session(fileroot, sessiondf, session, includeUnsortedSpikes=False, verbose=False):
    
    tempdf = sessiondf.loc[sessiondf['session'] == session]
    animal = tempdf.iloc[0].animal
    month = tempdf.iloc[0].month
    day = tempdf.iloc[0].day
    ctx = tempdf.iloc[0].task

    exp_kws = dict(fileroot = fileroot,
               animal = animal,
               session = session,
               month = month,
               day = day,
               ctx = ctx,
               includeUnsortedSpikes=includeUnsortedSpikes, # should be True for MUA analysis!
               verbose = verbose)
    
    return exp_kws

def get_sequence_df(binned_spikes, seq_bounds, seqds, seq_bins, ds, stype='MUA'):
    numSequences = len(seq_bins)
    numSpikes = np.zeros(numSequences, dtype=int)
    numCells = np.zeros(numSequences, dtype=int)
    for ii, (L, R) in enumerate(seq_bins):
        numSpikes[ii] = binned_spikes['data'][L:R+1].sum()
        numCells[ii] = np.count_nonzero(binned_spikes['data'][L:R+1].sum(axis=0))    

    ## hacky safety check that bin boundaries stay within range:
    if len(seq_bins[:,1][seq_bins[:,1] >= len(binned_spikes['data'])]) > 0:
        print('bin index has to be trimmed!')
        seq_bins[:,1][seq_bins[:,1] >= len(binned_spikes['data'])] = seq_bins[:,1][seq_bins[:,1] >= len(binned_spikes['data'])] - 1
    
    datadict = dict()
    datadict['animal'] = exp_kws['animal']
    datadict['month'] = exp_kws['month']
    datadict['day'] = exp_kws['day']
    datadict['session'] = exp_kws['session']
    datadict['ctx'] = exp_kws['ctx']
    datadict['iBdryL'] = seq_bounds[:,0]
    datadict['iBdryR'] = seq_bounds[:,1]
    datadict['ds'] = seqds
    datadict['iBBdryL'] = seq_bins[:,0]
    datadict['iBBdryR'] = seq_bins[:,1]
    datadict['bds'] = ds
    datadict['duration'] = (seq_bounds[:,1] - seq_bounds[:,0]) * seqds
    datadict['type'] = stype
    datadict['numSpikes'] = numSpikes
    datadict['numCells'] = numCells
    
    columns=['animal','month', 'day', 'session','ctx',
                                'iBdryL','iBdryR','ds', 'duration',
                                'iBBdryL','iBBdryR','bds',
                                'numSpikes', 'numCells',
                                'avgSpeed',
                                'type',
                                'notes']
        
    return pd.DataFrame(datadict, columns=columns)

def collapse_spike_train(st_array, verbose=False):
    # collapse a spike train array down to a single vector containing all spike times, but no unit info
    allspiketimes = np.array([],dtype=np.int64)
    for cc in st_array:
        allspiketimes = np.concatenate((allspiketimes,cc))
        
    allspiketimes.sort()
    
    if verbose:
        print ('collapsing {0} spike trains to a single spike train containing {1} spikes'.format(len(st_array),len(allspiketimes)))
    
    return allspiketimes

def bin_single_ordered_spike_list(st_array, fs, ds, verbose=False):
    num_bins = int(np.ceil(st_array.max()/fs/ds))
    maxtime = num_bins*ds

    spks_bin = np.zeros((num_bins,1))

    if verbose:
        print("binning data into {0} x {1:2.1f} ms temporal bins...".format(num_bins, ds*1000))

    # count number of spikes in an interval:
    spks_bin, bins = np.histogram(st_array/fs, bins=num_bins, density=False, range=(0,maxtime))
    
    bin_cntrs = np.arange(ds/2,num_bins*ds,ds)

    return spks_bin / 1, bins, bin_cntrs

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

def get_MUA_rate(spikes, ds=0.001, sigma=0.01, bw=6, verbose=False):
    import scipy.ndimage.filters

     # first, we collapse spike times into a single array:
    allspiketimes = collapse_spike_train(spikes.data, verbose=verbose)

    # then bin all spikes into small time bins:
    spks_bin, _, bin_cntrs = bin_single_ordered_spike_list(allspiketimes, spikes.samprate, ds=ds, verbose=verbose)

    # finally, smooth spikes with (sigma = 10 ms)-Gaussian with full bandwidth = 6*sigma (3*sigma half bandwidth)
    sigma = sigma / ds
    MUA = scipy.ndimage.filters.gaussian_filter1d(spks_bin, sigma, truncate=bw)
    
    return MUA

def get_sequence_events(x, PrimaryThreshold=None, SecondaryThreshold=None ):
    
    if PrimaryThreshold is None: # by default, threshold is 3 SDs above mean of x
        PrimaryThreshold = np.mean(x) + 3*np.std(x)
    
    if SecondaryThreshold is None: # by default, revert back to mean of x
        SecondaryThreshold = np.mean(x) # + 0*np.std(x)
        
    events, _ = find_threshold_crossing_events(x, PrimaryThreshold)
    
    # Find periods where value is > SecondaryThreshold; note that the previous periods should be within these!
    assert SecondaryThreshold <= PrimaryThreshold, "Secondary Threshold by definition should include more data than Primary Threshold"

    bounds, broader_maxes = find_threshold_crossing_events(x, SecondaryThreshold)

    # Find corresponding big windows for potential mua events
    #  Specifically, look for closest left edge that is just smaller
    outer_boundary_indices = np.searchsorted(bounds[:,0], events[:,0]);
    #  searchsorted finds the index after, so subtract one to get index before
    outer_boundary_indices = outer_boundary_indices - 1;

    # Find extended boundaries for mua events by pairing to larger windows
    #   (Note that there may be repeats if the larger window contains multiple > 3SD sections)
    bounds = bounds[outer_boundary_indices,:]
    maxes = broader_maxes[outer_boundary_indices]

    # Now, since all that we care about are the larger windows, so we should get rid of repeats
    _, unique_idx = np.unique(bounds[:,0], return_index=True)
    bounds = bounds[unique_idx,:] # SecondaryThreshold to SecondaryThreshold
    maxes = maxes[unique_idx]     # maximum MUA value during event
    events = events[unique_idx,:] # PrimaryThreshold to PrimaryThreshold
    
    return bounds, maxes, events

def match_bounds_with_bins(bounds, bds, ds):
    
    bbounds = np.zeros(bounds.shape, dtype=int)
    bbounds[:,0] = np.floor((bounds*(bds/ds))[:,0]).astype(int)
    bbounds[:,1] = np.ceil((bounds*(bds/ds))[:,1]).astype(int)
    
    return bbounds
    
##################################
#       plot spike rasters       #
##################################

def setdiff(first, second):
        second = set(second)
        return [item for item in first if item not in second]

def get_spikes_in_slice(spikes, L, R, ds, pindex=None):
    fs = spikes.samprate
    spks = []
    for cellSpikes in spikes['data']:
        cellSpikes = cellSpikes[cellSpikes / fs >= L*ds]
        cellSpikes = cellSpikes[cellSpikes / fs <= R*ds]
        spks.append(cellSpikes / fs)
        
    # rearrange cells to see sequential structure better:
    numCells = len(spikes['data'])
    if pindex is not None:
        npindex = setdiff(list(np.arange(numCells)), pindex)
        spksc = np.array(spks)
        spks = list(spksc[pindex])
        spks.extend(list(spksc[npindex]))
    return spks

##################################
#     smooth speed estimates     #
##################################

def butter_lowpass(cutoff, fs, order=5):
    from scipy.signal import butter
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filtfilt(data, cutoff, fs, order=5):
    from scipy.signal import filtfilt
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data, padlen=150)
    return y

def smooth_speed(posdf, cutoff=0.5, verbose=False):
    fs = posdf['fps'][0]
    dx = np.ediff1d(posdf['x'],to_begin=0)
    dy = np.ediff1d(posdf['y'],to_begin=0)
    dvdt = np.sqrt(np.square(dx) + np.square(dy))*fs # units (cm?) per second
    
    t0 = 0
    tend = len(dvdt)/fs # end in seconds

    dvdtlowpass = np.fmax(0,butter_lowpass_filtfilt(dvdt, cutoff=cutoff, fs=fs, order=6))

    if verbose:
        print('the animal ran an average of {0:2.2f} units/s'.format(dvdt.mean()))
        
    return dvdtlowpass

def find_closest(A, target):
    #A must be sorted
    idx = A.searchsorted(target)
    idx = np.clip(idx, 1, len(A)-1)
    left = A[idx-1]
    right = A[idx]
    idx -= target - left < right - target
    return idx

def get_position(posdf, time):
    # return position using linear interpolation
    # time can be a vector, in which case a vector of positions is returned
    
    try:
        if len(time) > 0:
            time = np.array(time)
            xp = pos[session]['time'][(pos[session]['time'] >time.min()) & (pos[session]['time'] <time.max()) ].values
            fpx = pos[session]['x'][(pos[session]['time'] >time.min()) & (pos[session]['time'] <time.max()) ].values
            x = np.interp(time, xp=xp, fp=fpx)
            fpy = pos[session]['y'][(pos[session]['time'] >time.min()) & (pos[session]['time'] <time.max()) ].values
            y = np.interp(time, xp=xp, fp=fpy)
    except:
        tidx = find_closest(posdf.time, time)[0]
        ctime = pos[session]['time'].iloc[tidx]
        if ctime < time:
            xp = [pos[session]['time'].iloc[tidx], pos[session]['time'].iloc[tidx+1]]
            fpx = [pos[session]['x'].iloc[tidx], pos[session]['x'].iloc[tidx+1]]
            fpy = [pos[session]['y'].iloc[tidx], pos[session]['y'].iloc[tidx+1]]
        else:
            xp = [pos[session]['time'].iloc[tidx-1], pos[session]['time'].iloc[tidx]]
            fpx = [pos[session]['x'].iloc[tidx-1], pos[session]['x'].iloc[tidx]]
            fpy = [pos[session]['y'].iloc[tidx], pos[session]['y'].iloc[tidx+1]]
        x = np.interp(time, xp=xp, fp=fpx)
        y = np.interp(time, xp=xp, fp=fpy)
    
    return x, y

def get_speed(posdf, time):
    try:
        if len(time) > 0:
            time = np.array(time)
            xp = pos[session]['time'][(pos[session]['time'] >time.min()) & (pos[session]['time'] <time.max()) ].values
            fp = pos[session]['speed'][(pos[session]['time'] >time.min()) & (pos[session]['time'] <time.max()) ].values
            speed = np.interp(time, xp=xp, fp=fp)
    except:
        tidx = find_closest(posdf.time, time)[0]
        ctime = pos[session]['time'].iloc[tidx]
        if ctime < time:
            xp = [pos[session]['time'].iloc[tidx], pos[session]['time'].iloc[tidx+1]]
            fp = [pos[session]['speed'].iloc[tidx], pos[session]['speed'].iloc[tidx+1]]
        else:
            xp = [pos[session]['time'].iloc[tidx-1], pos[session]['time'].iloc[tidx]]
            fp = [pos[session]['speed'].iloc[tidx-1], pos[session]['speed'].iloc[tidx]]
            
        speed = np.interp(time, xp=xp, fp=fp)
    
    return speed, speed.mean()
    