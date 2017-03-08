#encoding : utf-8
"""This file contains the nelpy io functions.

This entire module will probably be deprecated soon, so don't rely
on any of this to keep working!

Example
=======

datadirs = ['/home/etienne/Dropbox/neoReader/Data',
            'C:/etienne/Dropbox/neoReader/Data',
            'C:/Users/etien/Dropbox/neoReader/Data',
            '/Users/etienne/Dropbox/neoReader/Data',
            'D:/Dropbox/neoReader/Data']

fileroot = next( (dir for dir in datadirs if os.path.isdir(dir)), None)

if fileroot is None:
    raise FileNotFoundError('datadir not found')

exp_data = dict()
myhmm = dict()
data = dict()
sessiondate = dict()

sessions = ['session1', 'session2']

animal = 'gor01'; month,day = (6,7); sessiondate['session1'] = '11-26-53'; sessiondate['session2'] = '16-40-19' # 91 units, but session one has missing position data
# animal = 'gor01'; month,day = (6,12); sessiondate['session1'] = '15-55-31'; sessiondate['session2'] = '16-53-46' # 55 units
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

    exp_data[session]['spikes'] = nel.load_hc3_data(datatype='spikes', fs=32552, **exp_kws)
#     exp_data[session]['eeg'] = load_data(datatype='eeg', channels=[0,1,2], fs=1252, starttime=0, **exp_kws)
#     exp_data[session]['posdf'] = load_data(datatype='pos',**exp_kws)
#     exp_data[session]['speed'] = klab.get_smooth_speed(exp_data[session]['posdf'],fs=60,th=8,cutoff=0.5,showfig=False,verbose=False)

# make st1 and st2 explicitly available:
st1 = exp_data['session1']['spikes']
st2 = exp_data['session2']['spikes']


"""

__all__ = ['load_hc3_data']

import os.path
import pandas as pd
import numpy as np
import re
from .objects import *

# from mymap import Map

def get_num_electrodes(sessiondir):
    numelec = 0
    files = [f for f in os.listdir(sessiondir) if (os.path.isfile(os.path.join(sessiondir, f)))]
    for ff in files:
        try:
            found = re.search('\.clu\.[0-9]+$', ff).group(0)
            numelec+=1
        except:
            found=''
    if numelec > 0:
        return numelec
    else:
        raise ValueError('number of electrodes (shanks) could not be established...')

#datatype = ['spikes', 'eeg', 'pos', '?']
def load_hc3_data(fileroot, animal='gor01', year=2006, month=6, day=7, session='11-26-53', datatype='spikes', channels='all', fs=32552,starttime=0, ctx=None, verbose=False, includeUnsortedSpikes=False):

    fileroot = os.path.normpath(fileroot)
    anim_prefix = "{}-{}-{}".format(animal,month,day)
    session_prefix = "{}-{}-{}_{}".format(year,month,day,session)
    sessiondir = "{}/{}/{}".format(fileroot, anim_prefix, session_prefix)

    if (datatype=='spikes'):
        # NOTE: st_array[0] always corresponds to unsortable spikes (not mechanical noise). However, when includeUnsortedSpikes==True, then it gets populated
        #       with spike times; else, it just remains an empty list []

        #filename = "{}/{}/{}/{}.clu.1".format(fileroot, anim_prefix, session_prefix, session_prefix)
        filename = "{}/{}/{}/{}".format(fileroot, anim_prefix, session_prefix, session_prefix)
        #print(filename)
        if verbose:
            print("Loading data for session in directory '{}'...".format(sessiondir))
        num_elec = get_num_electrodes(sessiondir)
        if verbose:
            print('Number of electrode (.clu) files found:', num_elec)
        if includeUnsortedSpikes:
            st_array = [[]]
        else:
            st_array = []
        # note: using pandas.read_table is orders of magnitude faster here than using numpy.loadtxt
        for ele in np.arange(num_elec):
            #%time dt1a = np.loadtxt( base_filename1 + '.clu.' + str(ele + 1), skiprows=1,dtype=int)
            eudf = pd.read_table( filename + '.clu.' + str(ele + 1), header=None, names='u' ) # read unit numbers within electrode
            tsdf = pd.read_table( filename + '.res.' + str(ele + 1), header=None, names='t' ) # read sample numbers for spikes
            max_units = eudf.u.values[0]

            eu = eudf.u.values[1:]
            ts = tsdf.t.values
            # discard units labeled as '0' or '1', as these correspond to mechanical noise and unsortable units
            ts = ts[eu!=0]  # always discard mechanical noise
            eu = eu[eu!=0]  # always discard mechanical noise

            if not includeUnsortedSpikes:
                ts = ts[eu!=1]  # potentially discard unsortable spikes
                eu = eu[eu!=1]  # potentially discard unsortable spikes

            for uu in np.arange(max_units-2):
                st_array.append(ts[eu==uu+2])

            if includeUnsortedSpikes:
                st_array[0] = np.append(st_array[0], ts[eu==1])   # unit 0 now corresponds to unsortable spikes

        if verbose:
            print('Spike times (in sample numbers) for a total of {} units were read successfully...'.format(len(st_array)))

        if includeUnsortedSpikes:
            unit_ids = np.arange(len(st_array))
        else:
            unit_ids = np.arange(1, len(st_array)+1)

        # make sure that spike times are sorted! (this is not true for unit 0 of the hc-3 dataset, for example):
        for unit, spikes in enumerate(st_array):
            st_array[unit] = np.sort(spikes)

        spikes = SpikeTrainArray(st_array, label=session_prefix, fs=fs, unit_ids=unit_ids)

        # spikes = Map()
        # spikes['data'] = st_array
        # spikes['num_electrodes'] = num_elec
        # spikes['num_units'] = len(st_array)
        # spikes['samprate'] = fs
        # spikes['session'] = session_prefix

        return spikes

        ## continue from here... we want to keep cells that are inactive in some, but not all environments...
        # hence when extracting info, we must take all sessions in a recording day into account, and not just a specific recording session

    elif (datatype=='eeg'):
        filename = "{}/{}/{}/{}.eeg".format(fileroot, anim_prefix, session_prefix, session_prefix)
        if verbose:
            print("Loading EEG data from file '{}'".format(filename))
        num_elec = get_num_electrodes(sessiondir)
        num_channels = num_elec*8
        if channels=='all':
            channels = list(range(0,num_channels))
        if verbose:
            print('Number of electrode (.clu) files found: {}, with a total of {} channels'.format(num_elec, num_channels))
        dtype = np.dtype([(('ch' + str(ii)), 'i2')  for ii in range(num_channels) ])
        # read eeg data:
        try:
            eegdata = np.fromfile(filename, dtype=dtype, count=-1)
        except:
            print( "Unexpected error:", sys.exc_info()[0] )
            raise
        num_records = len(eegdata)
        if verbose:
            print("Successfully read {} samples for each of the {} channel(s).".format(num_records, len(channels)))

        data_arr = eegdata.astype(dtype).view('i2')
        data_arr = data_arr.reshape(num_records,num_channels)
        eeg = AnalogSignalArray(np.transpose(data_arr[:,channels]), fs=fs)
        eeg._metahc3channels = channels
        eeg._metahc3session = session_prefix
        # eeg['data'] = data_arr[:,channels]
        # eeg['channels'] = channels
        # eeg['samprate'] = fs
        # eeg['starttime'] = starttime
        # eeg['session'] = session_prefix

        return eeg

    elif (datatype=='pos'):
        filename = "{}/{}/{}/{}.whl".format(fileroot, anim_prefix, session_prefix, session_prefix)
        print("reading {} Hz position data from '{}'".format(fs, filename))
        dfwhl = pd.read_table(filename,sep='\t', skiprows=0, names=['x1', 'y1', 'x2', 'y2'] )
        dfwhl['x'] = (dfwhl['x1'] + dfwhl['x2']) / 2
        dfwhl['y'] = (dfwhl['y1'] + dfwhl['y2']) / 2
        dfwhl['fps'] = fs
        time = np.linspace(0,len(dfwhl)/fs,len(dfwhl)+1)
        dfwhl['time'] = time[:-1]
        return dfwhl
    else:
        raise ValueError('datatype is not handled')

def get_recording_days_for_animal(fileroot, animal):
    return [name for name in os.listdir(fileroot) if (os.path.isdir(os.path.join(fileroot, name))) & (name[0:len(animal)]==animal)]

def get_sessions_for_recording_day(fileroot, day):
    fileroot = os.path.join(fileroot,day)
    return [session for session in os.listdir(fileroot) if (os.path.isdir(os.path.join(fileroot, session)))]

def get_sessions(fileroot, animal='gor01', verbose=True):
    sessiondf = pd.DataFrame(columns=('animal','month','day','session','task'))
    fileroot = os.path.normpath(fileroot)
    if verbose:
        print("reading recording sessions for animal '{}' in directory '{}'...\n".format(animal,fileroot))
    for day in get_recording_days_for_animal(fileroot, animal):
        mm,dd = day.split('-')[1:]
        anim_prefix = "{}-{}-{}".format(animal,mm,dd)
        shortday = '-'.join([mm,dd])
        for session in get_sessions_for_recording_day(fileroot, day):
            infofile = "{}/{}/{}/{}.info".format(fileroot, anim_prefix, session, session)
            descr = ''
            try:
                with open(infofile, 'r') as f:
                    line = f.read()
                    if line.split('=')[0].strip()=='task':
                        descr = line.split('=')[-1].strip()
                if (descr == '') and (verbose == True):
                    print('Warning! Session type could not be established...')
            except IOError as e:
                print("I/O error({0}): {1}".format(e.errno, e.strerror))
            except ValueError:
                print ("Could not convert data to an integer.")
            except:
                print ("Unexpected error:", sys.exc_info()[0])
                raise
            session_hhmmss = session.split('_')[-1]
            # sessiondf = sessiondf.append(pd.DataFrame({'animal':[animal],'day':[shortday],'session':[session_hhmmss],'task':[descr]}),ignore_index=True)
            sessiondf = sessiondf.append(pd.DataFrame({'animal':[animal],'month':[mm], 'day':[dd],'session':[session_hhmmss],'task':[descr]}),ignore_index=True)
    if verbose:
        print(sessiondf)
    return sessiondf
