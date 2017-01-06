## load data

datadirs = ['/home/etienne/Dropbox/neoReader/Data',
            'C:/etienne/Dropbox/neoReader/Data',
            'C:/Users/etien/Dropbox/neoReader/Data',
            '/Users/etienne/Dropbox/neoReader/Data']

fileroot = next( (dir for dir in datadirs if os.path.isdir(dir)), None)

animal = 'gor01'; month,day = (6,7); session = '16-40-19' # 91 units

spikes = load_data(fileroot=fileroot, datatype='spikes',animal=animal, session=session, month=month, day=day, fs=32552, verbose=False)
eeg = load_data(fileroot=fileroot, datatype='eeg', animal=animal, session=session, month=month, day=day,channels=[0,1,2], fs=1252, starttime=0, verbose=False)
posdf = load_data(fileroot=fileroot, datatype='pos',animal=animal, session=session, month=month, day=day, verbose=False)
speed = klab.get_smooth_speed(posdf,fs=60,th=8,cutoff=0.5,showfig=False,verbose=False)

## bin ALL spikes
ds = 0.125 # bin spikes into 125 ms bins (theta-cycle inspired)
binned_spikes = klab.bin_spikes(spikes.data, ds=ds, fs=spikes.samprate, verbose=True)

## identify boundaries for running (active) epochs and then bin those observations into separate sequences:
runbdries = klab.get_boundaries_from_bins(eeg.samprate,bins=speed.active_bins,bins_fs=60)
binned_spikes_bvr = klab.bin_spikes(spikes.data, fs=spikes.samprate, boundaries=runbdries, boundaries_fs=eeg.samprate, ds=ds)

## stack data for hmmlearn:
seq_stk_bvr = sq.data_stack(binned_spikes_bvr, verbose=True)
seq_stk_all = sq.data_stack(binned_spikes, verbose=True)

## split data into train, test, and validation sets:
tr_b,vl_b,ts_b = sq.data_split(seq_stk_bvr, tr=50, vl=3, ts=50, randomseed = 0, verbose=True)

## train HMM on active behavioral data; training set (with a fixed, arbitrary number of states for now):
myhmm = sq.hmm_train(tr_b, num_states=35, n_iter=50, verbose=False)

llBVRtrain = np.array(list(sq.hmm_eval(myhmm, tr_b.data, symbol_by_symbol=True)))
llBVRtest = np.array(list(sq.hmm_eval(myhmm, ts_b.data, symbol_by_symbol=True)))
llALL = np.array(list(sq.hmm_eval(myhmm, seq_stk_all.data ,symbol_by_symbol=True)))

petal_bin_data = seq_stk_all.data.copy()
petal_ll = llALL.copy()

def petalplot(bindata, lldata, bins=150):
    from matplotlib.colors import LogNorm
    tmp = bindata.copy()
    tmp[bindata>0] = 1
    num_active_cells = tmp.sum(axis=1)
    my_cmap = plt.cm.get_cmap('Spectral')
    my_cmap.set_bad(alpha = 0.0)
    plt.hist2d(x=lldata,y=num_active_cells, bins=[bins, num_active_cells.max()+1], norm=LogNorm(), cmap=my_cmap, vmin=1 )
    cb = plt.colorbar()
    plt.xlabel('log probability of single time bin')
    plt.ylabel('number of active cells per bin')
    cb.set_label("bin count")
