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

# animal = 'gor01'; month,day = (6,7); session1 = '11-26-53'; session2 = '16-40-19' # 91 units, but session one has missing position data
animal = 'gor01'; month,day = (6,12); sessiondate['session1'] = '15-55-31'; sessiondate['session2'] = '16-53-46' # 55 units; this session has missing position data

for session in sessions:
    
    exp_data[session] = dict()
    
    exp_kws = dict(fileroot = fileroot,
               animal = animal,
               session = sessiondate[session],
               month = month,
               day = day,
               verbose = False)
    exp_data[session]['spikes'] = load_data(datatype='spikes', fs=32552, **exp_kws)
    exp_data[session]['eeg'] = load_data(datatype='eeg', channels=[0,1,2], fs=1252, starttime=0, **exp_kws)
    exp_data[session]['posdf'] = load_data(datatype='pos',**exp_kws)
    exp_data[session]['speed'] = klab.get_smooth_speed(exp_data[session]['posdf'],fs=60,th=8,cutoff=0.5,showfig=False,verbose=False)
    
    spikes = exp_data[session]['spikes']
    eeg = exp_data[session]['eeg']
    posdf = exp_data[session]['posdf']
    speed = exp_data[session]['speed']

    ## bin ALL spikes
    ds = 0.125 # bin spikes into 125 ms bins (theta-cycle inspired)
    binned_spikes_all = klab.bin_spikes(spikes.data, ds=ds, fs=spikes.samprate, verbose=True)

    ## identify boundaries for running (active) epochs and then bin those observations into separate sequences:
    runbdries = klab.get_boundaries_from_bins(eeg.samprate,bins=speed.active_bins,bins_fs=60)
    binned_spikes_bvr = klab.bin_spikes(spikes.data, fs=spikes.samprate, boundaries=runbdries, boundaries_fs=eeg.samprate, ds=ds)

    ## stack data for hmmlearn:
    seq_stk_bvr = sq.data_stack(binned_spikes_bvr, verbose=True)
    seq_stk_all = sq.data_stack(binned_spikes_all, verbose=True)

    ## split data into train, test, and validation sets:
    tr_b,vl_b,ts_b = sq.data_split(seq_stk_bvr, tr=50, vl=3, ts=50, randomseed = 0, verbose=False)

    data[session] = dict()
    data[session]['tr_b'] = tr_b
    data[session]['vl_b'] = vl_b
    data[session]['ts_b'] = ts_b
    
    num_states = 35
    myhmm[session] = sq.hmm_train(tr_b, num_states=num_states, n_iter=50, verbose=False)

#==========================================================

## fix data to test, and test in both models, but vary the length
data_to_test = data['session1']['ts_b']

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
            s1.append(myhmm['session1'].score(obs))
            s2.append(myhmm['session2'].score(obs))
        err_rates[sslen-1,mm] = (np.count_nonzero(np.array(s1) - np.array(s2) <0)) / len(s1)
    
    plt.plot(np.arange(maxlen)+1, err_rates[:,mm])

err_rates_ts1 = err_rates.copy()

## fix data to test, and test in both models, but vary the length
data_to_test = data['session2']['ts_b']

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
            s1.append(myhmm['session1'].score(obs))
            s2.append(myhmm['session2'].score(obs))
        err_rates[sslen-1,mm] = (np.count_nonzero(np.array(s1) - np.array(s2) >0)) / len(s1)
    
    plt.plot(np.arange(maxlen)+1, err_rates[:,mm])

err_rates_ts2 = err_rates.copy()

## fix data to test, and test in both models, but vary the length
data_to_test = data['session1']['tr_b']

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
            s1.append(myhmm['session1'].score(obs))
            s2.append(myhmm['session2'].score(obs))
        err_rates[sslen-1,mm] = (np.count_nonzero(np.array(s1) - np.array(s2) <0)) / len(s1)
    
    plt.plot(np.arange(maxlen)+1, err_rates[:,mm])

err_rates_tr1 = err_rates.copy()

## fix data to test, and test in both models, but vary the length
data_to_test = data['session2']['tr_b']

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
            s1.append(myhmm['session1'].score(obs))
            s2.append(myhmm['session2'].score(obs))
        err_rates[sslen-1,mm] = (np.count_nonzero(np.array(s1) - np.array(s2) >0)) / len(s1)
    
    plt.plot(np.arange(maxlen)+1, err_rates[:,mm])

err_rates_tr2 = err_rates.copy()

# saveData([err_rates_ts1, err_rates_ts2, err_rates_tr1, err_rates_tr2], 'err_rates_bvr', overwrite=True)