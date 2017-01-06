from mymap import Map

datadirs = ['/home/etienne/Dropbox/neoReader/Data',
            'C:/etienne/Dropbox/neoReader/Data',
            'C:/Users/etien/Dropbox/neoReader/Data',
            '/Users/etienne/Dropbox/neoReader/Data']

fileroot = next( (dir for dir in datadirs if os.path.isdir(dir)), None)

exp_data = dict()
myhmm = dict()
data = dict()
sessiondate = dict()

sessions = ['session1', 'session2']

# animal = 'gor01'; month,day = (6,7); sessiondate['session1'] = '11-26-53'; sessiondate['session2'] = '16-40-19' # 91 units, but session one has missing position data
animal = 'gor01'; month,day = (6,12); sessiondate['session1'] = '15-55-31'; sessiondate['session2'] = '16-53-46' # 55 units; this session has missing position data

for session in sessions:
    
    exp_data[session] = dict()
    
    exp_kws = dict(fileroot = fileroot,
               animal = animal,
               session = sessiondate[session],
               month = month,
               day = day,
               includeUnsortedSpikes=True, # should be True for MUA analysis!
               verbose = False)
    exp_data[session]['spikes'] = load_data(datatype='spikes', fs=32552, **exp_kws)
    exp_data[session]['eeg'] = load_data(datatype='eeg', channels=[0,1,2], fs=1252, starttime=0, **exp_kws)
    exp_data[session]['posdf'] = load_data(datatype='pos',**exp_kws)
    exp_data[session]['speed'] = klab.get_smooth_speed(exp_data[session]['posdf'],fs=60,th=8,cutoff=0.5,showfig=False,verbose=False)

def extract_subsequences_from_binned_spikes(binned_spikes, bins):
    data = spikes.data.copy()
    boundaries = klab.get_continuous_segments(bins)
    
    binned = Map()
    binned['bin_width'] = binned_spikes.bin_width
    binned['data'] = binned_spikes.data[bins,:]
    binned['boundaries'] = boundaries
    binned['boundaries_fs'] = 1/binned_spikes.bin_width   
    binned['sequence_lengths'] = (boundaries[:,1] - boundaries[:,0] + 1).flatten()
    
    return binned

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


def get_speed_during_MUA_event(bounds, mua_FS, speed):
    # returns the average running speed for each MUA event
    sbl = np.round(bounds[:,0] / mua_FS * speed.samprate).astype(np.int) # speed [index] bounds left
    sbr = np.round(bounds[:,1] / mua_FS * speed.samprate).astype(np.int) # speed [index] bounds right

    mua_speeds = (speed.data[sbl[ii]:sbr[ii]].mean() for ii in np.arange(bounds.shape[0]))

    return mua_speeds
    
def bin_and_stack_spikes_from_mua_events(st_array, ds, fs, mua_bounds, mua_bounds_fs, verbose=False):
    # bin and stack spikes from st_array (spikes.data) within the boundaries set by mua_bounds
    # mua_bounds is a Nx2 array of [start, stop] boundaries in mua_bounds_fs sample numbers
    
    # TODO: this is pretty slow when we provide boundaries. However, we can trivially parallelize the computation across units
    
    if verbose:
        print('binning {0} mua-event spike sequences into time bins of width {1} ms...'.format(len(st_array), ds*1000))
        
    binned_spikes = klab.bin_spikes(st_array, ds=ds, fs=fs, boundaries=mua_bounds, boundaries_fs=mua_bounds_fs ,verbose=verbose)
    stkd = sq.data_stack(binned_spikes)
    return stkd

data_swr = dict()
myhmm_swr = dict()

for session in sessions:
    
    spikes = exp_data[session]['spikes']
    eeg = exp_data[session]['eeg']
    posdf = exp_data[session]['posdf']
    speed = exp_data[session]['speed']

    # first, we collapse spike times into a single array:
    allspiketimes = collapse_spike_train(spikes.data, verbose=True)

    # then bin all spikes into small time bins:
    ds = 0.001 # 1 ms bins
    spks_bin, _, bin_cntrs = bin_single_ordered_spike_list(allspiketimes, spikes.samprate, ds=ds, verbose=True)

    # finally, smooth spikes with (sigma = 10 ms)-Gaussian with full bandwidth = 6*sigma (3*sigma half bandwidth)
    import scipy.ndimage.filters

    sigma = 0.010 / ds
    bw = 6
    smoothed_spikes = scipy.ndimage.filters.gaussian_filter1d(spks_bin, sigma, truncate=bw)
    
    threshold = np.mean(smoothed_spikes) + 3*np.std(smoothed_spikes)
    mua_events, _ = find_threshold_crossing_events(smoothed_spikes, threshold)
    
    # Find periods where value is > SecondaryThreshold; note that the previous periods should be within these!
    SecondaryThreshold = np.mean(smoothed_spikes) + 0 * np.std(smoothed_spikes)
    assert SecondaryThreshold < threshold, "Secondary Threshold by definition should include more data than Primary Threshold"

    mua_bounds, broader_maxes = find_threshold_crossing_events(smoothed_spikes, SecondaryThreshold)

    # Find corresponding big windows for potential mua events
    #  Specifically, look for closest left edge that is just smaller
    outer_boundary_indices = np.searchsorted(mua_bounds[:,0], mua_events[:,0]);
    #  searchsorted finds the index after, so subtract one to get index before
    outer_boundary_indices = outer_boundary_indices - 1;

    # Find extended boundaries for mua events by pairing to larger windows
    #   (Note that there may be repeats if the larger window contains multiple > 3SD sections)
    mua_bounds = mua_bounds[outer_boundary_indices,:]
    mua_maxes = broader_maxes[outer_boundary_indices]

    # Now, since all that we care about are the larger windows, so we should get rid of repeats
    _, unique_idx = np.unique(mua_bounds[:,0], return_index=True)
    mua_bounds = mua_bounds[unique_idx,:]
    mua_maxes = mua_maxes[unique_idx]
    mua_events = mua_events[unique_idx,:]
    
    LengthCriteria = 0.050 # 50 ms
    FS = 1 / ds
    mua_events = mua_events[mua_bounds[:,1] - mua_bounds[:,0] >= np.round(FS*LengthCriteria),:]
    mua_bounds = mua_bounds[mua_bounds[:,1] - mua_bounds[:,0] >= np.round(FS*LengthCriteria),:]

    print('{} MUA events detected'.format(len(mua_events)))

    print('WARNING!! Maximum event duration not yet handled... (but min event duration is now {0} ms)'.format(1000*LengthCriteria))

    MUALenghts = mua_bounds[:,1] - mua_bounds[:,0]
    
    mua_speeds = get_speed_during_MUA_event(mua_bounds, FS, speed)
    mua_speeds = np.array(list(mua_speeds))
    
    mua_bounds_fs = 1/ds
    upsec = 2; # maximum speed (in units per sec, upsec) for mua events

    mua_bounds_subset = mua_bounds[mua_speeds <= upsec] # restrict MUA events to only those where the animal was running slower than 1 units/s
    MUALenghts_subset = MUALenghts[mua_speeds <= upsec]
    
    print('{} MUA events remaining after velocity threshold'.format(len(MUALenghts_subset)))

    # TODO: this is pretty slow when we provide boundaries. However, we can trivially parallelize the computation across units
    ds = 0.01 # 10 ms bins
    mua_seq_stk_1ms_upsec = bin_and_stack_spikes_from_mua_events(st_array=spikes.data, ds=ds, fs=spikes.samprate, mua_bounds=mua_bounds_subset, mua_bounds_fs=mua_bounds_fs, verbose=True)
    
    data_swr[session] = dict()
    data_swr[session]['all'] = mua_seq_stk_1ms_upsec

    tr_swr,vl_swr,ts_swr = sq.data_split(data_swr[session]['all'], tr=50, vl=3, ts=50, randomseed = 0, verbose=True)

    # data_swr = dict()
    # data_swr[session] = dict()
    # data_swr[session]['all'] = seq_stk_swr
    data_swr[session]['tr'] = tr_swr
    data_swr[session]['vl'] = vl_swr
    data_swr[session]['ts'] = ts_swr

    num_states = 25
    myhmm_swr[session] = sq.hmm_train(data_swr[session]['tr'], num_states=num_states, n_iter=50, verbose=True)


#==========================================================

## fix data to test, and test in both models, but vary the length
data_to_test = data_swr['session1']['ts']

myStackedDataSeq = data_to_test.data.copy()
myStackedSeqLengths = data_to_test.sequence_lengths.copy()

seqlimits = np.cumsum(np.array([0] + list(myStackedSeqLengths)))

M = 1e1 # outer loop, number of times to compute entire curve --- used for error bars

N = 1e3 # number of subsequences to sample per sequence length

maxlen = myStackedSeqLengths.max()
# sslen = 1 # subseqeunce length

err_rates = np.zeros((maxlen,M))

for mm in np.arange(M):
    
    for sslen in np.arange(maxlen):
        sslen+=1
        s1 = []
        s2 = []
    #     print(sslen)
        for nn in np.arange(N):
            ssidx = np.random.choice(np.where(myStackedSeqLengths >= sslen)[0])
            ssstart = np.random.choice(myStackedSeqLengths[ssidx] - sslen + 1)

            obs = myStackedDataSeq[seqlimits[ssidx] + ssstart:seqlimits[ssidx] + ssstart + sslen,:]
            s1.append(myhmm_swr['session1'].score(obs))
            s2.append(myhmm_swr['session2'].score(obs))
        err_rates[sslen-1,mm] = (np.count_nonzero(np.array(s1) - np.array(s2) <0)) / len(s1)
    
    plt.plot(np.arange(maxlen)+1, err_rates[:,mm])

err_swr_ts1 = err_rates.copy()

## fix data to test, and test in both models, but vary the length
data_to_test = data_swr['session2']['ts']

myStackedDataSeq = data_to_test.data.copy()
myStackedSeqLengths = data_to_test.sequence_lengths.copy()

seqlimits = np.cumsum(np.array([0] + list(myStackedSeqLengths)))

M = 1e1 # outer loop, number of times to compute entire curve --- used for error bars

N = 1e3 # number of subsequences to sample per sequence length

maxlen = myStackedSeqLengths.max()
# sslen = 1 # subseqeunce length

err_rates = np.zeros((maxlen,M))

for mm in np.arange(M):
    
    for sslen in np.arange(maxlen):
        sslen+=1
        s1 = []
        s2 = []
    #     print(sslen)
        for nn in np.arange(N):
            ssidx = np.random.choice(np.where(myStackedSeqLengths >= sslen)[0])
            ssstart = np.random.choice(myStackedSeqLengths[ssidx] - sslen + 1)

            obs = myStackedDataSeq[seqlimits[ssidx] + ssstart:seqlimits[ssidx] + ssstart + sslen,:]
            s1.append(myhmm_swr['session1'].score(obs))
            s2.append(myhmm_swr['session2'].score(obs))
        err_rates[sslen-1,mm] = (np.count_nonzero(np.array(s1) - np.array(s2) >0)) / len(s1)
    
    plt.plot(np.arange(maxlen)+1, err_rates[:,mm])

err_swr_ts2 = err_rates.copy()

## fix data to test, and test in both models, but vary the length
data_to_test = data_swr['session1']['tr']

myStackedDataSeq = data_to_test.data.copy()
myStackedSeqLengths = data_to_test.sequence_lengths.copy()

seqlimits = np.cumsum(np.array([0] + list(myStackedSeqLengths)))

M = 1e1 # outer loop, number of times to compute entire curve --- used for error bars

N = 1e3 # number of subsequences to sample per sequence length

maxlen = myStackedSeqLengths.max()
# sslen = 1 # subseqeunce length

err_rates = np.zeros((maxlen,M))

for mm in np.arange(M):
    
    for sslen in np.arange(maxlen):
        sslen+=1
        s1 = []
        s2 = []
    #     print(sslen)
        for nn in np.arange(N):
            ssidx = np.random.choice(np.where(myStackedSeqLengths >= sslen)[0])
            ssstart = np.random.choice(myStackedSeqLengths[ssidx] - sslen + 1)

            obs = myStackedDataSeq[seqlimits[ssidx] + ssstart:seqlimits[ssidx] + ssstart + sslen,:]
            s1.append(myhmm_swr['session1'].score(obs))
            s2.append(myhmm_swr['session2'].score(obs))
        err_rates[sslen-1,mm] = (np.count_nonzero(np.array(s1) - np.array(s2) <0)) / len(s1)
    
    plt.plot(np.arange(maxlen)+1, err_rates[:,mm])

err_swr_tr1 = err_rates.copy()

## fix data to test, and test in both models, but vary the length
data_to_test = data_swr['session2']['tr']

myStackedDataSeq = data_to_test.data.copy()
myStackedSeqLengths = data_to_test.sequence_lengths.copy()

seqlimits = np.cumsum(np.array([0] + list(myStackedSeqLengths)))

M = 1e1 # outer loop, number of times to compute entire curve --- used for error bars

N = 1e3 # number of subsequences to sample per sequence length

maxlen = myStackedSeqLengths.max()
# sslen = 1 # subseqeunce length

err_rates = np.zeros((maxlen,M))

for mm in np.arange(M):
    
    for sslen in np.arange(maxlen):
        sslen+=1
        s1 = []
        s2 = []
    #     print(sslen)
        for nn in np.arange(N):
            ssidx = np.random.choice(np.where(myStackedSeqLengths >= sslen)[0])
            ssstart = np.random.choice(myStackedSeqLengths[ssidx] - sslen + 1)

            obs = myStackedDataSeq[seqlimits[ssidx] + ssstart:seqlimits[ssidx] + ssstart + sslen,:]
            s1.append(myhmm_swr['session1'].score(obs))
            s2.append(myhmm_swr['session2'].score(obs))
        err_rates[sslen-1,mm] = (np.count_nonzero(np.array(s1) - np.array(s2) >0)) / len(s1)
    
    plt.plot(np.arange(maxlen)+1, err_rates[:,mm])

err_swr_tr2 = err_rates.copy()

# saveData([err_swr_ts1, err_swr_ts2, err_swr_tr1, err_swr_tr2], 'err_rates_swr', overwrite=True)