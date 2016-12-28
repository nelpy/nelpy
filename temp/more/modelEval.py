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
animal = 'gor01'; month,day = (6,12); sessiondate['session1'] = '15-55-31'; sessiondate['session2'] = '16-53-46' # 55 units
# animal = 'gor01'; month,day = (6,13); sessiondate['session1'] = '14-42-6'; sessiondate['session2'] = '15-22-3'

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
    data[session]['all_b'] = seq_stk_bvr
    data[session]['tr_b'] = tr_b
    data[session]['vl_b'] = vl_b
    data[session]['ts_b'] = ts_b
    
    num_states = 35
    myhmm[session] = sq.hmm_train(tr_b, num_states=num_states, n_iter=50, verbose=False)

## fix data to test, and test in both models

meval = dict()

for session in sessions:
    meval[session] = dict()
    data_to_test = data[session]['ts_b']

    myStackedDataSeq = data_to_test.data.copy()
    myStackedSeqLengths = data_to_test.sequence_lengths.copy()

    s1 = []
    s2 = []

    seqlimits = np.cumsum(np.array([0] + list(myStackedSeqLengths)))
    for ee in np.arange(0,len(myStackedSeqLengths)):
        obs = myStackedDataSeq[seqlimits[ee]:seqlimits[ee+1],:]
        s1.append(myhmm['session1'].score(obs))
        s2.append(myhmm['session2'].score(obs))
        
    pdf, _ = np.histogram(myStackedSeqLengths, bins=myStackedSeqLengths.max()+2, range=(0,myStackedSeqLengths.max()+1))
    
    meval[session]['data'] = myStackedDataSeq
    meval[session]['seqLengths'] = myStackedSeqLengths
    meval[session]['s1'] = np.array(s1)
    meval[session]['s2'] = np.array(s2)
    meval[session]['pdf'] = pdf

# saveData(meval,'mevalday2')

a = meval['session1']['pdf']; b = meval['session2']['pdf']
if len(a) < len(b):
    c = b.copy()
    c[:len(a)] += a
else:
    c = a.copy()
    c[:len(b)] += b
    
maxlen = max([len(a), len(b)])
plt.step(np.arange(maxlen)+0.5,c,lw=2)

diffs1 = meval['session1']['s1'] - meval['session1']['s2']
diffs2 = meval['session2']['s1'] - meval['session2']['s2']
plt.plot(meval['session1']['seqLengths'], diffs1, 'ok', label='Y=ts1')
plt.plot(meval['session2']['seqLengths'], diffs2, 'or', label='Y=ts2')
plt.ylabel('log P(Y|env=1) - log P(Y|env=2)')
plt.xlabel('sequence length (# bins)')
plt.hlines(y=0, xmin=0, xmax=maxlen, linestyles='dashed')
plt.xlim([0, maxlen])
plt.legend()