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
animal = 'gor01'; month,day = (6,12); sessiondate['session1'] = '15-55-31'; sessiondate['session2'] = '16-53-46' # 55 units
# animal = 'gor01'; month,day = (6,13); sessiondate['session1'] = '14-42-6'; sessiondate['session2'] = '15-22-3'

for session in sessions:
    
    exp_data[session] = dict()
    
    exp_kws = dict(fileroot = fileroot,
               animal = animal,
               session = sessiondate[session],
               month = month,
               day = day,
               includeUnsortedSpikes=False, # should be True for MUA analysis!
               verbose = False)
    exp_data[session]['spikes'] = load_data(datatype='spikes', fs=32552, **exp_kws)
    exp_data[session]['eeg'] = load_data(datatype='eeg', channels=[0,1,2], fs=1252, starttime=0, **exp_kws)
    exp_data[session]['posdf'] = load_data(datatype='pos',**exp_kws)
    exp_data[session]['speed'] = klab.get_smooth_speed(exp_data[session]['posdf'],fs=60,th=8,cutoff=0.5,showfig=False,verbose=False)
    