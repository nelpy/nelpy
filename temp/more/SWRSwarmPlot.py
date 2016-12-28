## load data

datadirs = ['/home/etienne/Dropbox/neoReader/Data',
            'C:/etienne/Dropbox/neoReader/Data',
            'C:/Users/etien/Dropbox/neoReader/Data',
            '/Users/etienne/Dropbox/neoReader/Data']

fileroot = next( (dir for dir in datadirs if os.path.isdir(dir)), None)

animal = 'gor01'; month,day = (6,7); session = '16-40-19' # 91 units

# spikes = load_data(fileroot=fileroot, datatype='spikes',animal=animal, session=session, month=month, day=day, fs=32552, verbose=False)
eeg = load_data(fileroot=fileroot, datatype='eeg', animal=animal, session=session, month=month, day=day,channels=[0,1,2], fs=1252, starttime=0, verbose=False)
posdf = load_data(fileroot=fileroot, datatype='pos',animal=animal, session=session, month=month, day=day, verbose=False)
# speed = klab.get_smooth_speed(posdf,fs=60,th=8,cutoff=0.5,showfig=False,verbose=False)

ch = 2 # choose eeg (LFP) channel for ripple detection
data = eeg.data[:,ch].astype(int) # convert to 32 or 64 bit ints from 16 bit ints
fs_eeg = eeg.samprate

ripples = klab.detect_ripples(data, FS=fs_eeg, ThresholdSigma=3, SecondaryThresholdSigma=0, LengthCriteria=0.015)

## display SWR events overlayed on trajectory

StartTime = eeg.starttime
TimeAxis = StartTime + np.array(range(len(data))) / fs_eeg
ripple_centers = (ripples.ripple_bounds[:,1]/2 + ripples.ripple_bounds[:,0]/2) / fs_eeg + StartTime
# note: ripple_centers in seconds
exp_duration = TimeAxis[-1] # experimental duration in seconds
ripple_centers1 = ripple_centers[ripple_centers < exp_duration / 2]
ripple_centers2 = ripple_centers[ripple_centers > exp_duration / 2]
TimeAxis60Hz = StartTime + np.array(range(len(posdf.index))) / 60
RipIdx1 = np.searchsorted(TimeAxis60Hz, ripple_centers1)
RipIdx2 = np.searchsorted(TimeAxis60Hz, ripple_centers2)

_, swarmpts1 = sns.swarmplot(x=posdf.x1[RipIdx1], orient='h', size=3, color='r')
_, swarmpts2 = sns.swarmplot(x=posdf.x1[RipIdx2], orient='h', size=3, color='k')
