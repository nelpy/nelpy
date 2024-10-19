# encoding : utf-8
# IO module for reading data from CRCNS hc-18
# Author: Etienne Ackermann
# Date: November 2018

"""This file contains nelpy io functions for reading data from hc-18.

The data set includes 5 sessions (i.e., 5 rats), each consisting of three travel
sessions (passive, active, passive; 2×~15 laps each) interspersed with sleep
sessions (~90 min each).

See https://crcns.org/data-sets/hc/hc-18/about-hc-18.

datatype = ['spikes', 'lfp', 'pos', 'waveforms', 'metadata']

"""

__all__ = ["load_hc18_data"]

import logging
import os.path
import glob
import pandas as pd
import numpy as np
import re
import natsort as ns

import xmltodict

from abc import ABC, abstractmethod
from collections import OrderedDict

from ..core import *


class DataLoader(ABC):

    def __init__(self, basedir=None):

        if basedir is None:
            basedir = os.getcwd()

        basedir = os.path.normpath(basedir)
        self._basedir = basedir

    def __repr__(self):
        pass

    def __str__(self):
        pass

    def set_params(self, **params):
        pass

    def get_params(self, **params):
        pass

    @abstractmethod
    def load(self, *args, **kwargs):
        pass


class DataLoaderHC18(DataLoader):

    def __init__(self, basedir=None):
        super().__init__(basedir=basedir)

        self._sessions = None
        self._session_dirs = None
        self.data = OrderedDict()

        self.basedir = basedir

    def _get_sessions(self, path=None, cache=True):
        if path is None:
            path = self.basedir  # + '/data/'
        path = os.path.normpath(path)
        if not os.path.isdir(path):
            logging.error('"{}" should be a directory'.format(path))
            raise FileNotFoundError('"{}" should be a directory'.format(path))

        sessions = next(os.walk(path))[1]
        if len(sessions) == 0:
            logging.error('No sessions found at "{}"'.format(path))
            raise FileNotFoundError('No sessions found at "{}"'.format(path))

        dirs = [os.path.normpath(path + "/" + session + "/") for session in sessions]

        sortorder = np.argsort(sessions)
        sessions = np.array(sessions)[sortorder].tolist()
        dirs = np.array(dirs)[sortorder].tolist()
        if cache:
            self._sessions = sessions
            self._session_dirs = dirs

        return sessions, dirs

    @property
    def basedir(self):
        if self._basedir is None:
            raise ValueError("basedir has not been initialized yet!")
        return self._basedir

    @basedir.setter
    def basedir(self, path):
        if path is None:
            return
        path = os.path.normpath(path)
        if not os.path.isdir(path):
            logging.error('"{}" should be a directory'.format(path))
            raise FileNotFoundError('"{}" should be a directory'.format(path))
        self._basedir = path
        self._load_xml()
        self._drop_xml()
        self._get_sessions()

    @property
    def sessions(self):
        if self._sessions is None:
            _, _ = self._get_sessions()
        return self._sessions

    @property
    def session_dirs(self):
        if self._session_dirs is None:
            _, _ = self._get_sessions()
        return self._session_dirs

    def _get_num_probes(self, path):
        """Determine the number of probes (silicon probe shanks, tetrodes, ...).

        Parameters
        ----------

        Returns
        -------
        dict with {session: num_probes}

        """

        num = 0
        files = [f for f in os.listdir(path) if (os.path.isfile(os.path.join(path, f)))]
        for ff in files:
            try:
                found = re.search("\.clu\.[0-9]+$", ff).group(0)
                logging.info('Found {} in session at "{}"'.format(found, path))
                num += 1
            except Exception:
                pass
        if num == 0:
            logging.warning(
                'Did not find any .clu files for session "{}"'.format(session)
            )

        return num

    # def _get_num_probes(self, path=None):
    #     """Determine the number of probes (silicon probe shanks, tetrodes, ...).

    #     Parameters
    #     ----------

    #     Returns
    #     -------
    #     dict with {session: num_probes}

    #     """
    #     if path is None:
    #         if self._sessions is None:
    #             self._get_sessions()
    #         sessions, dirs = self._sessions, self._session_dirs
    #     else:
    #         sessions, dirs = self._get_sessions(path=path, cache=False)

    #     nums = {}
    #     for (sdir, session) in zip(dirs, sessions):
    #         num = 0
    #         files = [f for f in os.listdir(sdir) if (os.path.isfile(os.path.join(sdir, f)))]
    #         for ff in files:
    #             try:
    #                 found = re.search('\.clu\.[0-9]+$', ff).group(0)
    #                 logging.info('Found {} in session "{}"'.format(found, session))
    #                 num+=1
    #             except Exception:
    #                 pass
    #         if num == 0:
    #             logging.warning('Did not find any .clu files for session "{}"'.format(session))
    #         nums[session] = num
    #     return nums

    def _load_xml(self):
        sessions, dirs = self.sessions, self.session_dirs

        for session in sessions:
            p = self.data.get(session, None)
            if p is None:
                self.data[session] = {}
            p = self.data[session].get("params", None)
            if p is None:
                self.data[session]["params"] = {}

        for session, sdir in zip(sessions, dirs):
            with open(sdir + "/" + session + ".xml") as fd:
                xml = xmltodict.parse(fd.read())
                self.data[session]["xml"] = xml
                self.data[session]["params"]["fs_wideband"] = float(
                    xml["parameters"]["acquisitionSystem"]["samplingRate"]
                )
                self.data[session]["params"]["n_channels"] = float(
                    xml["parameters"]["acquisitionSystem"]["nChannels"]
                )
                self.data[session]["params"]["fs_lfp"] = float(
                    xml["parameters"]["fieldPotentials"]["lfpSamplingRate"]
                )
                self.data[session]["params"]["fs_video"] = float(
                    xml["parameters"]["video"]["samplingRate"]
                )
                self.data[session]["params"]["video_width"] = float(
                    xml["parameters"]["video"]["width"]
                )
                self.data[session]["params"]["video_height"] = float(
                    xml["parameters"]["video"]["height"]
                )
                self.data[session]["params"]["snippet_len"] = int(
                    xml["parameters"]["neuroscope"]["spikes"]["nSamples"]
                )
                self.data[session]["params"]["snippet_peak_idx"] = int(
                    xml["parameters"]["neuroscope"]["spikes"]["peakSampleIndex"]
                )
                self.data[session]["params"]["n_probes"] = self._get_num_probes(
                    path=sdir
                )

    def _drop_xml(self):
        sessions = self.sessions
        for session in sessions:
            self.data[session].pop("xml", None)

    def _load_events(self):
        sessions, dirs = self.sessions, self.session_dirs

        for session, sdir in zip(sessions, dirs):
            events = OrderedDict()
            filename = "{}/{}.cat.evt".format(sdir, session)
            with open(filename) as f:
                for line in f:
                    line = line.strip()
                    timestamp, descr = line.split(" ", maxsplit=1)
                    descr = descr.split("-", maxsplit=3)[-1]
                    timestamp = float(timestamp) / 1000
                    try:
                        events[descr] = EpochArray((events[descr], timestamp))
                    except KeyError:
                        events[descr] = timestamp

            self.data[session]["events"] = events

        return

    def _load_pos(self):
        sessions, dirs = self.sessions, self.session_dirs

        for session, sdir in zip(sessions, dirs):
            fs = self.data[session]["params"]["fs_video"]
            width = self.data[session]["params"]["video_width"]
            height = self.data[session]["params"]["video_height"]

            if session in ["Train-292-20150501", "Train-314-20160118"]:
                width = 640
                height = 480

            filename = "{}/{}.pos".format(sdir, session)

            logging.info('reading position data from "{}"'.format(filename))
            posdata = np.array(
                pd.read_table(filename, sep="\t", skiprows=0, names=["x", "y"])
            )
            timestamps = np.linspace(0, len(posdata) / fs, len(posdata))

            logging.info("dropping samples with missing data")
            dropidx = np.where(posdata == -1)[0]
            posdata = np.delete(posdata, dropidx, 0)
            timestamps = np.delete(timestamps, dropidx, 0)

            logging.info("building nelpy PositionArray")
            pos = PositionArray(
                posdata.T,
                abscissa_vals=timestamps,
                fs=fs,
                xlim=[0, width],
                ylim=[0, height],
            )
            self.data[session]["pos"] = pos

    def _load_spikes(self, include_mua=False, includeWaveforms=False):

        # NOTE: st_array[0] always corresponds to unsortable spikes (i.e., MUA,
        # not mechanical noise). However, when includeUnsortedSpikes==True, then
        # it gets populated with spike times; else, it just remains an empty
        # list []

        sessions, dirs = self.sessions, self.session_dirs

        for session, sdir in zip(sessions, dirs):
            fs = self.data[session]["params"]["fs_wideband"]
            fs = 32552.083
            num_probes = self.data[session]["params"]["n_probes"]
            snippet_len = self.data[session]["params"]["snippet_len"]
            snippet_peak_idx = self.data[session]["params"]["snippet_peak_idx"]
            n_channels = int(self.data[session]["params"]["n_channels"] / 16)
            if n_channels == 4:
                print("tetrodes!")
            elif n_channels == 8:
                print("octrodes!")
            else:
                raise ValueError("unexpected number of channels for hc-18")

            clu_files = ns.humansorted(glob.glob(sdir + "/{}.clu*".format(session)))

            if include_mua:
                st_array = [[]]
                wf_array = [[]]
            else:
                st_array = []
                wf_array = []
            wfdt = np.dtype(
                "<h", (snippet_len, n_channels)
            )  # waveform datatype (.spk files)
            # note: using pandas.read_table is orders of magnitude faster here than using numpy.loadtxt
            for clu_file in clu_files:
                pre, post = clu_file.split("clu")
                res_file = pre + "res" + post
                spk_file = pre + "spk" + post
                # %time dt1a = np.loadtxt( base_filename1 + '.clu.' + str(probe + 1), skiprows=1,dtype=int)
                eudf = pd.read_table(
                    clu_file, header=None, names="u"
                )  # read unit numbers within electrode
                tsdf = pd.read_table(
                    res_file, header=None, names="t"
                )  # read sample numbers for spikes
                if includeWaveforms:
                    waveforms = np.fromfile(spk_file, dtype=wfdt)
                    waveforms = np.reshape(
                        waveforms,
                        (
                            int(len(waveforms) / (snippet_len * n_channels)),
                            snippet_len,
                            n_channels,
                        ),
                    )
                    waveforms = waveforms[:, snippet_peak_idx, :]

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

                if not include_mua:
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

                if include_mua:
                    st_array[0] = np.append(
                        st_array[0], ts[eu == 1]
                    )  # unit 0 now corresponds to unsortable spikes
                    if includeWaveforms:
                        if len(wf_array[0]) > 0:
                            wf_array[0] = np.vstack((wf_array[0], waveforms[eu == 1]))
                        else:
                            wf_array[0] = waveforms[eu == 1]

            if include_mua:
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
                spikes = SpikeTrainArray(st_array, fs=fs, unit_ids=unit_ids)
                spikes._marks = wf_array
            else:
                spikes = SpikeTrainArray(st_array, fs=fs, unit_ids=unit_ids)

            self.data[session]["spikes"] = spikes

    def load(self, datatype="all", data=None):
        """Load (and overwrite!) data into dictionary."""
        if data is None:
            data = self.data
        else:
            logging.error("passing in a data dictionary is not fully supported yet!")
            raise NotImplementedError

        if datatype == "pos":  # load position data
            self._load_pos()

        if datatype == "spikes":  # load sorted spike data
            self._load_spikes()

        if datatype == "events":
            self._load_events()


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
