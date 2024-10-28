# encoding : utf-8
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
sessiontime = dict()

sessions = ['session1', 'session2']

animal = 'gor01'; month,day = (6,7); sessiontime['session1'] = '11-26-53'; sessiontime['session2'] = '16-40-19' # 91 units, but session one has missing position data
# animal = 'gor01'; month,day = (6,12); sessiontime['session1'] = '15-55-31'; sessiontime['session2'] = '16-53-46' # 55 units
# animal = 'gor01'; month,day = (6,13); sessiontime['session1'] = '14-42-6'; sessiontime['session2'] = '15-22-3'

for session in sessions:

    exp_data[session] = dict()

    exp_kws = dict(fileroot = fileroot,
               animal = animal,
               session = sessiontime[session],
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

__all__ = ["load_hc3_data"]

import os.path
import sys
import pandas as pd
import numpy as np
import re
from ..core._analogsignalarray import (
    AnalogSignalArray,
)
from ..core._eventarray import (
    SpikeTrainArray,
)

# from mymap import Map


def get_num_electrodes(sessiondir, verbose=False):
    numelec = 0
    files = [
        f
        for f in os.listdir(sessiondir)
        if (os.path.isfile(os.path.join(sessiondir, f)))
    ]
    for ff in files:
        try:
            found = re.search("\.clu\.[0-9]+$", ff).group(0)
            if verbose:
                print(found)
            numelec += 1
        except Exception:
            found = ""
    if numelec > 0:
        return numelec
    else:
        raise ValueError("number of electrodes (shanks) could not be established...")


# datatype = ['spikes', 'eeg', 'pos', '?']
def load_hc3_data(
    fileroot,
    animal="gor01",
    year=2006,
    month=6,
    day=7,
    sessiontime="11-26-53",
    track=None,
    datatype="spikes",
    channels="all",
    fs=None,
    starttime=0,
    ctx=None,
    verbose=False,
    includeUnsortedSpikes=False,
    includeWaveforms=False,
):

    fileroot = os.path.normpath(fileroot)
    if track is None:
        anim_prefix = "{}-{}-{}".format(animal, month, day)
        session_prefix = "{}-{}-{}_{}".format(year, month, day, sessiontime)
        sessiondir = "{}/{}/{}".format(fileroot, anim_prefix, session_prefix)
    else:
        anim_prefix = "{}".format(animal)
        session_prefix = "{}-{}-{}_{}".format(
            year, month, str(day).zfill(2), sessiontime
        )
        sessiondir = "{}/{}/{}/{}".format(
            fileroot, anim_prefix, track, session_prefix
        )  # track can be 'one', 'two', or 'sleep'

    if datatype == "spikes":
        if fs is None:
            fs = 32552
        # NOTE: st_array[0] always corresponds to unsortable spikes (not mechanical noise). However, when includeUnsortedSpikes==True, then it gets populated
        #       with spike times; else, it just remains an empty list []

        # filename = "{}/{}/{}/{}.clu.1".format(fileroot, anim_prefix, session_prefix, session_prefix)
        filename = "{}/{}".format(sessiondir, session_prefix)
        # print(filename)
        if verbose:
            print("Loading data for session in directory '{}'...".format(sessiondir))
        num_elec = get_num_electrodes(sessiondir, verbose=verbose)
        if verbose:
            print("Number of electrode (.clu) files found:", num_elec)
        if includeUnsortedSpikes:
            st_array = [[]]
            wf_array = [[]]
        else:
            st_array = []
            wf_array = []
        wfdt = np.dtype("<h", (54, 8))  # waveform datatype (.spk files)
        # note: using pandas.read_table is orders of magnitude faster here than using numpy.loadtxt
        for ele in np.arange(num_elec):
            # %time dt1a = np.loadtxt( base_filename1 + '.clu.' + str(ele + 1), skiprows=1,dtype=int)
            eudf = pd.read_table(
                filename + ".clu." + str(ele + 1), header=None, names="u"
            )  # read unit numbers within electrode
            tsdf = pd.read_table(
                filename + ".res." + str(ele + 1), header=None, names="t"
            )  # read sample numbers for spikes
            if includeWaveforms:
                waveforms = np.fromfile(filename + ".spk." + str(ele + 1), dtype=wfdt)
                waveforms = np.reshape(
                    waveforms, (int(len(waveforms) / (54 * 8)), 54, 8)
                )
                waveforms = waveforms[:, 26, :]

            max_units = eudf.u.values[0]

            eu = eudf.u.values[1:]
            ts = tsdf.t.values

            if includeWaveforms:
                noise_idx = np.argwhere(eu == 0).squeeze()
                hash_idx = np.argwhere(eu == 1).squeeze()
                all_idx = set(np.arange(len(eu)))
                discard_idx = set(noise_idx)

            # discard units labeled as '0' or '1', as these correspond to mechanical noise and unsortable units
            ts = ts[eu != 0]  # always discard mechanical noise
            eu = eu[eu != 0]  # always discard mechanical noise

            if not includeUnsortedSpikes:
                ts = ts[eu != 1]  # potentially discard unsortable spikes
                eu = eu[eu != 1]  # potentially discard unsortable spikes
                if includeWaveforms:
                    discard_idx = discard_idx.union(set(hash_idx))

            if includeWaveforms:
                keep_idx = all_idx - discard_idx
                waveforms = waveforms[sorted(list(keep_idx))]

            for uu in np.arange(max_units - 2):
                st_array.append(ts[eu == uu + 2])
                if includeWaveforms:
                    wf_array.append(waveforms[eu == uu + 2])

            if includeUnsortedSpikes:
                st_array[0] = np.append(
                    st_array[0], ts[eu == 1]
                )  # unit 0 now corresponds to unsortable spikes
                if includeWaveforms:
                    if len(wf_array[0]) > 0:
                        wf_array[0] = np.vstack((wf_array[0], waveforms[eu == 1]))
                    else:
                        wf_array[0] = waveforms[eu == 1]

        if verbose:
            print(
                "Spike times (in sample numbers) for a total of {} units were read successfully...".format(
                    len(st_array)
                )
            )

        if includeUnsortedSpikes:
            unit_ids = np.arange(len(st_array))
        else:
            unit_ids = np.arange(1, len(st_array) + 1)

        # make sure that spike times are sorted! (this is not true for unit 0 of the hc-3 dataset, for example):
        for unit, spikes in enumerate(st_array):
            order = np.argsort(spikes)
            st_array[unit] = spikes[order] / fs
            if includeWaveforms:
                wf_array[unit] = wf_array[unit][order]

        if includeWaveforms:
            # spikes = MarkedSpikeTrainArray(st_array, marks=wf_array, label=session_prefix, fs=fs, unit_ids=unit_ids)
            spikes = SpikeTrainArray(
                st_array, label=session_prefix, fs=fs, unit_ids=unit_ids
            )
            spikes._marks = wf_array
        else:
            spikes = SpikeTrainArray(
                st_array, label=session_prefix, fs=fs, unit_ids=unit_ids
            )

        # spikes = Map()
        # spikes['data'] = st_array
        # spikes['num_electrodes'] = num_elec
        # spikes['num_units'] = len(st_array)
        # spikes['samprate'] = fs
        # spikes['session'] = session_prefix

        return spikes

        ## continue from here... we want to keep cells that are inactive in some, but not all environments...
        # hence when extracting info, we must take all sessions in a recording day into account, and not just a specific recording session

    if datatype == "clusterless":
        if fs is None:
            fs = 32552

        filename = "{}/{}".format(sessiondir, session_prefix)

        if verbose:
            print("Loading data for session in directory '{}'...".format(sessiondir))
        num_elec = get_num_electrodes(sessiondir, verbose=verbose)
        if verbose:
            print("Number of electrode (.clu) files found:", num_elec)
        st_array = []
        mark_array = []

        wfdt = np.dtype("<h", (54, 8))  # waveform datatype (.spk files)
        # note: using pandas.read_table is orders of magnitude faster here than using numpy.loadtxt
        for ele in np.arange(num_elec):
            # %time dt1a = np.loadtxt( base_filename1 + '.clu.' + str(ele + 1), skiprows=1,dtype=int)
            eudf = pd.read_table(
                filename + ".clu." + str(ele + 1), header=None, names="u"
            )  # read unit numbers within electrode
            tsdf = pd.read_table(
                filename + ".res." + str(ele + 1), header=None, names="t"
            )  # read sample numbers for spikes
            if verbose:
                print(len(tsdf), "spikes detected")
            try:
                waveforms = np.fromfile(filename + ".spk." + str(ele + 1), dtype=wfdt)
            except FileNotFoundError:
                print(
                    "could not find {}, skipping clusterless for this session".format(
                        filename + ".spk." + str(ele + 1)
                    )
                )
                return None, None
            if verbose:
                print(len(waveforms) / (54 * 8), "waveforms detected")
            if len(tsdf) - len(waveforms) / (54 * 8) != 0:
                print(
                    "could not find a one-to-one match between spike times and waveforms... skipping clusterless data for {}".format(
                        filename
                    )
                )
                return None, None
            waveforms = np.reshape(waveforms, (int(len(waveforms) / (54 * 8)), 54, 8))
            marks = waveforms[
                :, 26, :
            ]  # this assumed that spikes have been aligned to have peak at index 26

            max_units = eudf.u.values[0]

            eu = eudf.u.values[1:]
            ts = tsdf.t.values

            noise_idx = np.argwhere(eu == 0).squeeze()
            hash_idx = np.argwhere(eu == 1).squeeze()
            all_idx = set(np.arange(len(eu)))
            discard_idx = set(noise_idx)

            # discard units labeled as '0' or '1', as these correspond to mechanical noise and unsortable units
            ts = ts[eu != 0]  # always discard mechanical noise
            eu = eu[eu != 0]  # always discard mechanical noise

            if not includeUnsortedSpikes:
                ts = ts[eu != 1]  # potentially discard unsortable spikes
                eu = eu[eu != 1]  # potentially discard unsortable spikes
                discard_idx = discard_idx.union(set(hash_idx))

            keep_idx = all_idx - discard_idx
            marks = marks[sorted(list(keep_idx))]

            st_array.append(ts)
            mark_array.append(marks)

        if verbose:
            print(
                "Spike times and marks for a total of {} electrodes were read successfully...".format(
                    num_elec
                )
            )

        # make sure that spike times are sorted! (this is not true for unit 0 of the hc-3 dataset, for example):
        for ele, spikes in enumerate(st_array):
            order = np.argsort(spikes)
            st_array[ele] = spikes[order] / fs
            mark_array[ele] = mark_array[ele][order]

        return np.array(st_array), np.array(mark_array)

    elif datatype == "eeg":
        if fs is None:
            fs = 1252
        filename = "{}/{}.eeg".format(sessiondir, session_prefix)
        if verbose:
            print("Loading EEG data from file '{}'".format(filename))
        num_elec = get_num_electrodes(sessiondir)
        num_channels = num_elec * 8
        if channels == "all":
            channels = list(range(0, num_channels))
        if verbose:
            print(
                "Number of electrode (.clu) files found: {}, with a total of {} channels".format(
                    num_elec, num_channels
                )
            )
        dtype = np.dtype([(("ch" + str(ii)), "i2") for ii in range(num_channels)])
        # read eeg data:
        try:
            eegdata = np.fromfile(filename, dtype=dtype, count=-1)
        except:
            print("Unexpected error:", sys.exc_info()[0])
            raise
        num_records = len(eegdata)
        if verbose:
            print(
                "Successfully read {} samples for each of the {} channel(s).".format(
                    num_records, len(channels)
                )
            )

        data_arr = eegdata.astype(dtype).view("i2")
        data_arr = data_arr.reshape(num_records, num_channels)
        eeg = AnalogSignalArray(np.transpose(data_arr[:, channels]), fs=fs)
        eeg._metahc3channels = channels
        eeg._metahc3session = session_prefix
        # eeg['data'] = data_arr[:,channels]
        # eeg['channels'] = channels
        # eeg['samprate'] = fs
        # eeg['starttime'] = starttime
        # eeg['session'] = session_prefix

        return eeg

    elif datatype == "pos":
        if fs is None:
            fs = 60
        filename = "{}/{}.whl".format(sessiondir, session_prefix)
        if verbose:
            print("reading {} Hz position data from '{}'".format(fs, filename))
        dfwhl = pd.read_table(
            filename, sep="\t", skiprows=0, names=["x1", "y1", "x2", "y2"]
        )
        dfwhl["x"] = (dfwhl["x1"] + dfwhl["x2"]) / 2
        dfwhl["y"] = (dfwhl["y1"] + dfwhl["y2"]) / 2
        dfwhl["fps"] = fs
        time = np.linspace(0, len(dfwhl) / fs, len(dfwhl) + 1)
        dfwhl["time"] = time[:-1]
        return dfwhl
    elif datatype == "unit_map":
        # in each file, cluster 0 is discarded as mechanical noise;
        # in each filr, cluster 1 is unsorted spikes; these are all pooled into unit 0
        # the rest follows numerically...
        unit_map = {}
        filename = "{}/{}".format(sessiondir, session_prefix)
        if verbose:
            print("Loading data for session in directory '{}'...".format(sessiondir))
        num_elec = get_num_electrodes(sessiondir, verbose=verbose)
        if verbose:
            print("Number of electrode (.clu) files found:", num_elec)
        # note: using pandas.read_table is orders of magnitude faster here than using numpy.loadtxt
        unit_id = 1
        for ele in np.arange(num_elec):
            with open(filename + ".clu." + str(ele + 1)) as myfile:
                n_units = int(myfile.readline())

            for nn in range(2, n_units):
                unit_map[ele + 1, nn] = unit_id
                unit_id += 1
        return unit_map

    else:
        raise ValueError("datatype is not handled")


def get_recording_days_for_animal(fileroot, animal):
    return [
        name
        for name in os.listdir(fileroot)
        if (os.path.isdir(os.path.join(fileroot, name)))
        & (name[0 : len(animal)] == animal)
    ]


def get_sessions_for_recording_day(fileroot, day):
    fileroot = os.path.join(fileroot, day)
    return [
        session
        for session in os.listdir(fileroot)
        if (os.path.isdir(os.path.join(fileroot, session)))
    ]


def get_sessions(fileroot, animal="gor01", verbose=True):
    sessiondf = pd.DataFrame(columns=("animal", "month", "day", "session", "task"))
    fileroot = os.path.normpath(fileroot)
    if verbose:
        print(
            "reading recording sessions for animal '{}' in directory '{}'...\n".format(
                animal, fileroot
            )
        )
    for day in get_recording_days_for_animal(fileroot, animal):
        mm, dd = day.split("-")[1:]
        anim_prefix = "{}-{}-{}".format(animal, mm, dd)
        # shortday = "-".join([mm, dd])
        for session in get_sessions_for_recording_day(fileroot, day):
            infofile = "{}/{}/{}/{}.info".format(
                fileroot, anim_prefix, session, session
            )
            descr = ""
            try:
                with open(infofile, "r") as f:
                    line = f.read()
                    if line.split("=")[0].strip() == "task":
                        descr = line.split("=")[-1].strip()
                if (descr == "") and (verbose is True):
                    print("Warning! Session type could not be established...")
            except IOError as e:
                print("I/O error({0}): {1}".format(e.errno, e.strerror))
            except ValueError:
                print("Could not convert data to an integer.")
            except:
                print("Unexpected error:", sys.exc_info()[0])
                raise
            session_hhmmss = session.split("_")[-1]
            # sessiondf = sessiondf.append(pd.DataFrame({'animal':[animal],'day':[shortday],'session':[session_hhmmss],'task':[descr]}),ignore_index=True)
            sessiondf = sessiondf.append(
                pd.DataFrame(
                    {
                        "animal": [animal],
                        "month": [mm],
                        "day": [dd],
                        "session": [session_hhmmss],
                        "task": [descr],
                    }
                ),
                ignore_index=True,
            )
    if verbose:
        print(sessiondf)
    return sessiondf


def get_hc3_dataframe():
    """this is not for hc-3, but rather for the same (but modified) dataset
    from Kamran Diba. Essentially Kamran just has a different (much better) unit
    sorting than what's available on CRCNS. However, with the waveform snippets
    available on CRCNS, we can do clusterless or our own sorting and get the
    same performance as Kamran's private dataset.

    This dataframe is useful in giving a summary of what data is availalbe, as
    well as demarcations for long and short segments on the track, etc.

    The data was manually compiled by E Ackermann.
    """
    df = pd.DataFrame(
        columns=(
            "animal",
            "month",
            "day",
            "time",
            "track",
            "segments",
            "segment_labels",
            "whl",
            "n_cells",
            "Notes",
            "has_waveforms",
        )
    )

    df = df.append(
        {
            "animal": "pin01",
            "month": 11,
            "day": 1,
            "time": "12-58-54",
            "whl": True,
            "track": "unknown",
            "segment_labels": ("long", "short"),
            "segments": [(0, 1670), (2100, 3025)],
        },
        ignore_index=True,
    )
    df = df.append(
        {
            "animal": "pin01",
            "month": 11,
            "day": 3,
            "time": "11-0-53",
            "whl": False,
            "track": "unknown",
            "segment_labels": (),
            "segments": [],
        },
        ignore_index=True,
    )
    df = df.append(
        {
            "animal": "pin01",
            "month": 11,
            "day": 3,
            "time": "20-28-3",
            "whl": True,
            "track": "unknown",
            "segment_labels": ("long", "short"),
            "segments": [(0, 700), (720, 1080)],
        },
        ignore_index=True,
    )

    df = df.append(
        {
            "animal": "vvp01",
            "month": 4,
            "day": 9,
            "time": "17-29-30",
            "whl": True,
            "track": "one",
            "segment_labels": ("long", "short", "long2"),
            "segments": [(0, 800), (905, 1395), (1445, 1660)],
        },
        ignore_index=True,
    )
    df = df.append(
        {
            "animal": "vvp01",
            "month": 4,
            "day": 10,
            "time": "12-25-50",
            "whl": True,
            "track": "one",
            "segment_labels": ("long", "short"),
            "segments": [(0, 870), (970, 1390)],
        },
        ignore_index=True,
    )
    df = df.append(
        {
            "animal": "vvp01",
            "month": 4,
            "day": 10,
            "time": "21-2-40",
            "whl": True,
            "track": "one",
            "segment_labels": ("long", "short", "long2"),
            "segments": [(0, 590), (625, 937), (989, 1081)],
        },
        ignore_index=True,
    )
    df = df.append(
        {
            "animal": "vvp01",
            "month": 4,
            "day": 11,
            "time": "15-16-59",
            "whl": True,
            "track": "one",
            "segment_labels": ("long", "short"),
            "segments": [(0, 667), (734, 975)],
        },
        ignore_index=True,
    )
    df = df.append(
        {
            "animal": "vvp01",
            "month": 4,
            "day": 12,
            "time": "14-39-31",
            "whl": True,
            "track": "one",
            "segment_labels": ("long", "short"),
            "segments": [(0, 581), (620, 887)],
        },
        ignore_index=True,
    )
    df = df.append(
        {
            "animal": "vvp01",
            "month": 4,
            "day": 12,
            "time": "17-53-55",
            "whl": True,
            "track": "one",
            "segment_labels": ("long", "short", "long2"),
            "segments": [(0, 466), (534, 840), (888, 1178)],
        },
        ignore_index=True,
    )
    df = df.append(
        {
            "animal": "vvp01",
            "month": 4,
            "day": 16,
            "time": "15-12-23",
            "whl": True,
            "track": "one",
            "segment_labels": ("long", "short"),
            "segments": [(22, 528), (650, 997)],
        },
        ignore_index=True,
    )
    df = df.append(
        {
            "animal": "vvp01",
            "month": 4,
            "day": 17,
            "time": "12-33-47",
            "whl": True,
            "track": "one",
            "segment_labels": ("long", "short"),
            "segments": [(9, 438), (459, 865)],
        },
        ignore_index=True,
    )
    df = df.append(
        {
            "animal": "vvp01",
            "month": 4,
            "day": 18,
            "time": "13-6-1",
            "whl": True,
            "track": "one",
            "segment_labels": ("long", "short"),
            "segments": [(14, 496), (519, 795)],
        },
        ignore_index=True,
    )
    df = df.append(
        {
            "animal": "vvp01",
            "month": 4,
            "day": 18,
            "time": "15-23-32",
            "whl": True,
            "track": "one",
            "segment_labels": ("long", "short"),
            "segments": [(8, 283), (295, 499)],
        },
        ignore_index=True,
    )
    df = df.append(
        {
            "animal": "vvp01",
            "month": 4,
            "day": 19,
            "time": "13-34-40",
            "whl": True,
            "track": "one",
            "segment_labels": ("long", "short"),
            "segments": [(10, 394), (413, 657)],
        },
        ignore_index=True,
    )
    df = df.append(
        {
            "animal": "vvp01",
            "month": 4,
            "day": 19,
            "time": "16-48-9",
            "whl": True,
            "track": "one",
            "segment_labels": ("long", "short"),
            "segments": [(0, 271), (359, 601)],
        },
        ignore_index=True,
    )
    df = df.append(
        {
            "animal": "vvp01",
            "month": 4,
            "day": 21,
            "time": "10-24-35",
            "whl": True,
            "track": "one",
            "segment_labels": ("long", "short"),
            "segments": [(14, 465), (490, 777)],
        },
        ignore_index=True,
    )
    df = df.append(
        {
            "animal": "vvp01",
            "month": 4,
            "day": 25,
            "time": "14-28-51",
            "whl": True,
            "track": "one",
            "segment_labels": ("long", "short"),
            "segments": [(9, 394), (405, 616)],
        },
        ignore_index=True,
    )
    df = df.append(
        {
            "animal": "vvp01",
            "month": 4,
            "day": 25,
            "time": "17-17-6",
            "whl": True,
            "track": "one",
            "segment_labels": ("long", "short"),
            "segments": [(7, 316), (330, 520)],
        },
        ignore_index=True,
    )
    df = df.append(
        {
            "animal": "vvp01",
            "month": 4,
            "day": 26,
            "time": "13-22-13",
            "whl": True,
            "track": "one",
            "segment_labels": ("long", "short"),
            "segments": [(20, 375), (415, 614)],
        },
        ignore_index=True,
    )
    df = df.append(
        {
            "animal": "vvp01",
            "month": 4,
            "day": 27,
            "time": "14-43-12",
            "whl": True,
            "track": "one",
            "segment_labels": ("long", "top"),
            "segments": [(9, 611), (640, 908)],
        },
        ignore_index=True,
    )
    df = df.append(
        {
            "animal": "vvp01",
            "month": 4,
            "day": 28,
            "time": "12-17-27",
            "whl": True,
            "track": "one",
            "segment_labels": ("long", "bottom"),
            "segments": [(11, 433), (446, 677)],
        },
        ignore_index=True,
    )
    df = df.append(
        {
            "animal": "vvp01",
            "month": 4,
            "day": 28,
            "time": "16-48-29",
            "whl": True,
            "track": "one",
            "segment_labels": ("long", "bottom"),
            "segments": [(6, 347), (363, 600)],
        },
        ignore_index=True,
    )
    df = df.append(
        {
            "animal": "vvp01",
            "month": 4,
            "day": 9,
            "time": "16-40-54",
            "whl": True,
            "track": "two",
            "segment_labels": ("long", "short"),
            "segments": [(289, 1150), (1224, 1709)],
        },
        ignore_index=True,
    )
    df = df.append(
        {
            "animal": "vvp01",
            "month": 4,
            "day": 10,
            "time": "12-58-3",
            "whl": True,
            "track": "two",
            "segment_labels": ("long", "short"),
            "segments": [(21, 923), (984, 1450)],
        },
        ignore_index=True,
    )
    df = df.append(
        {
            "animal": "vvp01",
            "month": 4,
            "day": 10,
            "time": "19-11-57",
            "whl": True,
            "track": "two",
            "segment_labels": ("long", "short"),
            "segments": [(25, 916), (1050, 1477)],
        },
        ignore_index=True,
    )
    df = df.append(
        {
            "animal": "vvp01",
            "month": 4,
            "day": 11,
            "time": "12-48-38",
            "whl": True,
            "track": "two",
            "segment_labels": ("long", "short"),
            "segments": [(8, 705), (851, 1284)],
        },
        ignore_index=True,
    )
    df = df.append(
        {
            "animal": "vvp01",
            "month": 4,
            "day": 11,
            "time": "16-2-46",
            "whl": True,
            "track": "two",
            "segment_labels": ("long", "short"),
            "segments": [(10, 578), (614, 886)],
        },
        ignore_index=True,
    )
    df = df.append(
        {
            "animal": "vvp01",
            "month": 4,
            "day": 12,
            "time": "14-59-23",
            "whl": True,
            "track": "two",
            "segment_labels": ("long", "short"),
            "segments": [(7, 284)],
        },
        ignore_index=True,
    )
    df = df.append(
        {
            "animal": "vvp01",
            "month": 4,
            "day": 12,
            "time": "15-25-59",
            "whl": True,
            "track": "two",
            "segment_labels": ("long", "short"),
            "segments": [(13, 462), (498, 855)],
        },
        ignore_index=True,
    )
    df = df.append(
        {
            "animal": "vvp01",
            "month": 4,
            "day": 16,
            "time": "14-49-24",
            "whl": True,
            "track": "two",
            "segment_labels": ("long", "short"),
            "segments": [(14, 749), (773, 1035)],
        },
        ignore_index=True,
    )
    df = df.append(
        {
            "animal": "vvp01",
            "month": 4,
            "day": 16,
            "time": "18-47-52",
            "whl": True,
            "track": "two",
            "segment_labels": ("long", "short"),
            "segments": [(13, 433), (444, 752)],
        },
        ignore_index=True,
    )
    df = df.append(
        {
            "animal": "vvp01",
            "month": 4,
            "day": 17,
            "time": "12-52-15",
            "whl": True,
            "track": "two",
            "segment_labels": ("long", "short"),
            "segments": [(13, 464), (473, 801)],
        },
        ignore_index=True,
    )
    df = df.append(
        {
            "animal": "vvp01",
            "month": 4,
            "day": 18,
            "time": "13-28-57",
            "whl": True,
            "track": "two",
            "segment_labels": ("long", "short"),
            "segments": [(0, 396), (404, 619)],
        },
        ignore_index=True,
    )
    df = df.append(
        {
            "animal": "vvp01",
            "month": 4,
            "day": 18,
            "time": "15-38-2",
            "whl": True,
            "track": "two",
            "segment_labels": ("long", "short"),
            "segments": [(13, 307), (316, 510)],
        },
        ignore_index=True,
    )
    df = df.append(
        {
            "animal": "vvp01",
            "month": 4,
            "day": 19,
            "time": "13-50-7",
            "whl": True,
            "track": "two",
            "segment_labels": ("long", "short"),
            "segments": [(9, 297), (304, 505)],
        },
        ignore_index=True,
    )
    df = df.append(
        {
            "animal": "vvp01",
            "month": 4,
            "day": 19,
            "time": "16-37-40",
            "whl": True,
            "track": "two",
            "segment_labels": ("long", "short"),
            "segments": [(17, 279), (289, 467)],
        },
        ignore_index=True,
    )
    df = df.append(
        {
            "animal": "vvp01",
            "month": 4,
            "day": 21,
            "time": "11-19-2",
            "whl": True,
            "track": "two",
            "segment_labels": ("long", "short"),
            "segments": [(6, 358), (363, 577)],
        },
        ignore_index=True,
    )
    df = df.append(
        {
            "animal": "vvp01",
            "month": 4,
            "day": 25,
            "time": "13-20-55",
            "whl": True,
            "track": "two",
            "segment_labels": ("long", "short"),
            "segments": [(30, 334), (348, 569)],
        },
        ignore_index=True,
    )
    df = df.append(
        {
            "animal": "vvp01",
            "month": 4,
            "day": 25,
            "time": "17-33-28",
            "whl": True,
            "track": "two",
            "segment_labels": ("long", "short"),
            "segments": [(10, 277), (286, 456)],
        },
        ignore_index=True,
    )
    df = df.append(
        {
            "animal": "vvp01",
            "month": 4,
            "day": 26,
            "time": "13-51-50",
            "whl": True,
            "track": "two",
            "segment_labels": ("long", "short", "long2"),
            "segments": [(9, 317), (324, 506), (515, 766)],
        },
        ignore_index=True,
    )
    df = df.append(
        {
            "animal": "vvp01",
            "month": 4,
            "day": 27,
            "time": "18-21-57",
            "whl": True,
            "track": "two",
            "segment_labels": ("long", "top"),
            "segments": [(13, 279), (292, 493)],
        },
        ignore_index=True,
    )
    df = df.append(
        {
            "animal": "vvp01",
            "month": 4,
            "day": 28,
            "time": "12-38-13",
            "whl": True,
            "track": "two",
            "segment_labels": ("long", "bottom"),
            "segments": [(5, 286), (291, 526)],
        },
        ignore_index=True,
    )
    df = df.append(
        {
            "animal": "vvp01",
            "month": 4,
            "day": 28,
            "time": "17-6-14",
            "whl": True,
            "track": "two",
            "segment_labels": ("long", "bottom", "short"),
            "segments": [(8, 343), (350, 593), (617, 791)],
        },
        ignore_index=True,
    )

    df = df.append(
        {
            "animal": "gor01",
            "month": 6,
            "day": 7,
            "time": "11-26-53",
            "track": "one",
            "whl": True,
            "segment_labels": ("long"),
            "segments": [(0, 1730)],
            "Notes": "missing position data for short segment",
        },
        ignore_index=True,
    )
    df = df.append(
        {
            "animal": "gor01",
            "month": 6,
            "day": 7,
            "time": "16-40-19",
            "track": "two",
            "whl": True,
            "segment_labels": ("long", "short"),
            "segments": [(0, 1180), (1250, 2580)],
            "Notes": "there is a .whl_back file---what is this?",
        },
        ignore_index=True,
    )
    df = df.append(
        {
            "animal": "gor01",
            "month": 6,
            "day": 12,
            "time": "15-55-31",
            "track": "one",
            "whl": True,
            "segment_labels": ("long", "short"),
            "segments": [(0, 660), (710, 1120)],
        },
        ignore_index=True,
    )
    df = df.append(
        {
            "animal": "gor01",
            "month": 6,
            "day": 12,
            "time": "16-53-46",
            "track": "two",
            "whl": True,
            "segment_labels": ("long", "short"),
            "segments": [(0, 470), (490, 796)],
        },
        ignore_index=True,
    )
    df = df.append(
        {
            "animal": "gor01",
            "month": 6,
            "day": 13,
            "time": "14-42-6",
            "track": "one",
            "whl": True,
            "segment_labels": ("long", "short"),
            "segments": [(0, 520), (540, 845)],
        },
        ignore_index=True,
    )
    df = df.append(
        {
            "animal": "gor01",
            "month": 6,
            "day": 13,
            "time": "15-22-3",
            "track": "two",
            "whl": True,
            "segment_labels": ("long", "short"),
            "segments": [(0, 530), (540, 865)],
        },
        ignore_index=True,
    )
    df = df.append(
        {
            "animal": "gor01",
            "month": 6,
            "day": 8,
            "time": "15-46-47",
            "track": "two",
            "whl": True,
            "segment_labels": ("long"),
            "segments": [(0, 2400)],
            "Notes": "short segment seems bad",
        },
        ignore_index=True,
    )
    df = df.append(
        {
            "animal": "gor01",
            "month": 6,
            "day": 8,
            "time": "21-16-25",
            "track": "two",
            "whl": True,
            "segment_labels": ("long", "short"),
            "segments": [(0, 720), (750, 1207)],
        },
        ignore_index=True,
    )
    df = df.append(
        {
            "animal": "gor01",
            "month": 6,
            "day": 9,
            "time": "22-24-40",
            "track": "two",
            "whl": True,
            "segment_labels": ("long", "short"),
            "segments": [(0, 912), (920, 2540)],
        },
        ignore_index=True,
    )
    df = df.append(
        {
            "animal": "gor01",
            "month": 6,
            "day": 8,
            "time": "14-26-15",
            "track": "one",
            "whl": False,
            "segment_labels": (""),
            "segments": [],
        },
        ignore_index=True,
    )
    df = df.append(
        {
            "animal": "gor01",
            "month": 6,
            "day": 9,
            "time": "1-22-43",
            "track": "one",
            "whl": True,
            "segment_labels": ("long", "short"),
            "segments": [(0, 1012), (1035, 1652)],
        },
        ignore_index=True,
    )
    df = df.append(
        {
            "animal": "gor01",
            "month": 6,
            "day": 9,
            "time": "3-23-37",
            "track": "one",
            "whl": True,
            "segment_labels": ("long"),
            "segments": [(28, 530)],
            "Notes": "no short segment?",
        },
        ignore_index=True,
    )
    df = df.append(
        {
            "animal": "gor01",
            "month": 6,
            "day": 8,
            "time": "10-43-37",
            "track": "sleep",
            "whl": False,
            "segment_labels": ("sleep"),
            "segments": [],
        },
        ignore_index=True,
    )
    df = df.append(
        {
            "animal": "gor01",
            "month": 6,
            "day": 8,
            "time": "17-26-16",
            "track": "sleep",
            "whl": False,
            "segment_labels": ("sleep"),
            "segments": [],
        },
        ignore_index=True,
    )

    df.month = df.month.astype(np.int64)
    df.day = df.day.astype(np.int64)

    df[(df.animal == "gor01") & df.whl]

    return df
