__all__ = ['EventArray']

import warnings
import numpy as np 
import copy
import numbers

from ._epocharray import EpochArray

def fsgetter(self):
    """(float) [generic getter] Sampling frequency"""
    if self._fs is None:
        warnings.warn("No sampling frequency has been specified!")
    return self._fs

########################################################################
# class EventArray
########################################################################
class EventArray:
    """Stores time/timestamps (tdata) of events. EventArray is restricted in its
    support by EpochArray and contains an optional state variable in order to 
    represent particular states at corresponding timestamps (tdata).

    Parameters
    ----------
    tdata : np.array(dtype=np.float,dimension=N)
        Timestamps at which events occur. tdata can either be sample numbers or
        time but if it's in units of time be weary of fs and tdata_in_timstamps.
    
    state : np.array(dtype=np.float,dimension=N), optional
        State is to contain states associated with particular timestamps passed
        in from tdata (e.g. 0 or 1 associated with a correct or incorrect
        trigger). By default this is None. 
    
    support : EpochArray, optional
        EpochArray on which tdata is restricted. Default is to cover all of
        tdata.
    
    fs : float, scalar, optional
        Sampling rate in Hz. If fs is passed in as a parameter, by default, time
        is assumed to be in sample numbers unless tdata_in_samples is set to 
        False. As such, time attribute will be calculated by dividing tdata by
        fs passed in. See fs_acquisition if timestamps are stored at a different
        rate than what was sampled and marked by the system. By default this is 
        None.
    
    fs_acquisition : float, scalar, optional
        Optional to store sampling rate in Hz of the acquisition system. This 
        should be used when tdata is passed in timestamps associated with the 
        acquisition system but is stored in step sizes that are of a different
        sampling rate. E.g. times could be stamped at 30kHz but stored in a 
        decimated fashion at 3kHz so instead of 1,2,3,4,5,6,7,8,9,10,11,12,...,
        20,21,22,23,24,25,26,... it would be 1,10,20,30,40,50,60,70,80,90,100,
        110,120,...,200,210,220,230,240,250,260,... (get the idea?). In cases
        like this, fs_acquisition as opposed to fs should be used to calculate 
        time while fs should remain at the decimated sampling rate for further
        uses. Note, this is of more importance to ``AnalogSignalArray`` than 
        ``EventArray`` but still worth maintaining for consistency amongst 
        objects. Additionally, fs_acquisition can be used to just divide tdata
        without storing fs (not sure why anyone would want to do this though but
        feel free to do it before I remove this functionality!) By default this 
        is None.
    
    tdata_in_samples : bool, optional
        Boolean flag set to True by default. Determines if tdata is to be scaled
        by fs in order to generate time attribute.
    
    labels : np.array(dtype=np.str,dimension=N)
        Labeling each one of the states or timestamps. By default this is None 
        but all signals must be labeled if any are labeled or else we will label
        in order of signals passed in and replace the others with Nones. If more
        labels are passed in than signals, we will truncate the remaining! If we
        are nice (which we are for the most part), we will display a warning 
        upon doing any of these things! :P Lastly, it is worth noting that most
        logical and type error checking for this is expected to be done by the 
        user. Inputs are casted to strings and stored in a numpy array.
    
    empty : bool
        Return an empty ``EventArray`` if requested aka this flag is True. By 
        default this will be set to False cuz who in their right minds would 
        want to instantiate an empty ``EventArray``, amirite?

    Attributes
    ----------
    tdata : np.array
        See ``Parameters``

    time : np.array
        Converts tdata to units of time. time can be equal to tdata if fs is not
        specified or if fs is one or if tdata_in_timestamps is set to False.

    state : np.array 
        See ``Parameters``
    
    support : EpochArray, optional
        See ``Parameters``
    
    fs : float, scalar, optional
        See ``Parameters``
    
    fs_acquisition : float, scalar, optional
        See ``Parameters``. Set equal to fs if fs is provided and this is None.
    
    labels : np.array 
        See ``Parameters``
    """
    __attributes__ = ['_tdata','_time','_state','_support','_fs',\
                      '_fs_acquisition', '_labels']
    def __init__(self, tdata, *, state=None, support=None, fs=None, \
                 fs_acquisition=None, tdata_in_samples=True, labels=None, \
                 empty=None):

        #if empty object is requested, give it to 'em one!
        if empty:
            for attr in self.__attributes__:
                exec("self." + attr + " = None")
            self._support = EpochArray(empty=True)
            return
        np.concatenate
        #check if single Event or multiple events in array
        tdata = np.squeeze(tdata).astype(float)
        try:
            if tdata.shape[0] == tdata.size:
                tdata = np.array(tdata,ndmin=2).astype(float)
        except ValueError:
            raise TypeError("Unsupported type! Integer or floating point "
                            "expected for tdata")
        self._tdata = tdata #let's go ahead and just set tdata since we aren't
                            #going to need to change this later

        #check if state array was provided. This must _exactly_ match shape of 
        #tdata
        if state is not None:
            state = np.squeeze(state).astype(float)
            try:
                if state.shape[0] == state.size:
                    state = np.array(state,ndmin=2).astype(float)
            except ValueError:
                raise TypeError("Unsupported type! Integer or floating point "
                                "expected for state") 
            if state.shape[0] != tdata.shape[0]:
                raise IndexError("tdata and state dimensions must _exactly_"
                                 " match!")
            for i in range(0,tdata.shape[0]):
                if len(state[i][:]) != len(tdata[i][:]):
                    raise IndexError("tdata and state dimensions must _exactly_"
                                     " match!")
            
            self._state = state
        else:
            self._state = np.zeros([0,self._tdata.shape[0]])
            self._state[:] = np.NAN

        #handle labels
        if labels is not None:
            labels = np.asarray(labels,dtype=np.str)
            #label size doesn't match
            if labels.shape[0] > tdata.shape[0]:
                warnings.warn("More labels than tdata! labels are sliced to "
                              "size of tdata")
                labels = labels[0:tdata.shape[0]]
            elif labels.shape[0] < tdata.shape[0]:
                warnings.warn("Less labels than tdata! labels are filled with "
                              "Nones to match tdata shape")
                for i in range(labels.shape[0],tdata.shape[0]):
                    labels.append(None)
        self._labels = labels
            
        #handle fs and fs_acquisition
        self._fs = None
        self._fs_acquisition = None
        self.fs = fs
        if fs is not None:
            if fs_acquisition is not None:
                self.fs_acquisition = fs_acquisition
                if tdata_in_samples:
                    self._time = self.tdata / self.fs_acquisition
                else:
                    self._time = self.tdata
            else:
                self.fs_acquisition = fs
                if tdata_in_samples:
                    self._time = self.tdata / self.fs_acquisition
                else:
                    self._time = self.tdata
        elif fs_acquisition is not None:
            warnings.warn("Why are you setting fs_acquisition to scale tdata to"
                          " time as opposed to fs? Seriously, is there a reason"
                          " for this? Contact me (Shayok) if there is or I'll"
                          " probably remove this functionality!")
            self.fs_acquisition = fs_acquisition
            if tdata_in_samples:
                self._time = self.tdata / self.fs_acquisition
            else:
                self._time = self.tdata
        else:
            self._time = self.tdata

        #support restrictions
        if support is not None:
            self._restrict_to_epoch_array(epocharray=support)
            if self._support.isempty:
                warnings.warn("Support is empty. Empty EventArray returned with"
                              " specified sampling rate variables.")
        else:
            self._support = EpochArray([np.min(self._tdata),\
                                        np.max(self._tdata)],\
                                        fs=self._fs_acquisition)

    def _restrict_to_epoch_array(self, *, epocharray=None, update=True):
        """Restrict self._time and self._state to an EpochArray. If no 
        EpochArray is specified, self._support is used.

        Parameters
        ----------
        epocharray : EpochArray, optional
            EpochArray on which to restrict EventArray. Default is self._support
        update : bool, optional
            Overwrite self._support with epocharray if True (default)
        """
        if epocharray is None:
            epocharray = self._support

        try:
            if epocharray.isempty:
                warnings.warn("Support specified is empty")
                self._support = epocharray
                self._tdata = np.zeros([0,self._tdata.shape[0]])
                self._tdata[:] = np.NAN
                self._time = np.zeros([0,self._time.shape[0]])
                self._time[:] = np.NAN
                self._state = np.zeros([0,self._state.shape[0]])
                self._state[:] = np.NAN
                self._labels = None
                return
        except AttributeError:
            raise AttributeError("EpochArray expected")

        indices = []
        for eptime in epocharray.time:
            t_start = eptime[0]
            t_stop = eptime[1]
            indices.append((self._time >= t_start) & (self._time <= t_stop))
        indices = np.any(np.column_stack(indices), axis=1)
        if np.count_nonzero(indices) < len(self._time):
            warnings.warn("ignoring timestamps outside of support")
        try:
            self._tdata = self._tdata[:,indices]
        except IndexError:
            self._tdata = np.zeros([0,self._tdata.shape[0]])
            self._tdata[:] = np.NAN
            self._time = np.zeros([0,self._time.shape[0]])
            self._time[:] = np.NAN
            self._state = np.zeros([0,self._state.shape[0]])
            self._state = np.NAN
            self._labels = None
        self._time = self._time[indices]
        self._tdata = self._tdata[indices]
        if update:
            self._support = epocharray

    def __repr__(self):
        warnings.warn("Rethink this. EventArray repr will change!")
        address_str = " at " + str(hex(id(self)))
        if self.isempty:
            return "<empty EventArray" + address_str + ">"
        epstr = " ({} events)".format(self.n_events)
        try:
            if(self.n_arrays > 0):
                nstr = " %s event array(s)%s" % (self.n_arrays, epstr)
        except IndexError:
            nstr = " 1 event array%s" % epstr
        return "<EventArray%s:%s>" % (address_str, nstr)

    @property
    def n_events(self):
        """number of total events"""
        events = 0
        for x in range(0,self._tdata.shape[0]):
            events += len(self._tdata[x,:])
        return events

    @property
    def n_arrays(self):
        """number of total events"""
        return self._tdata.shape[0]
    
    @property
    def isempty(self):
        """(bool) checks length of tdata"""
        try:
            return len(self._tdata) == 0
        except TypeError: #TypeError should happen if _tdata == []
            return True

    @property
    def support(self):
        """(nelpy.EpochArray) The support of the underlying EventArray
        (in seconds).
        """
        return self._support

    @property
    def tdata(self):
        if self._tdata is None:
            warnings.warn("No tdata. The EventArray is empty?")
        return self._tdata

    @property
    def time(self):
        """(np.array N-D) Time calculated from tdata
        """
        if self._time is None:
            warnings.warn("No time Calculated. Is the EventArray empty?")
        return self._time

    @property
    def fs(self):
        """(float) Sampling Frequency"""
        return fsgetter(self)

    @fs.setter
    def fs(self, val):
        """(float) Set sampling rate"""
        if self._fs == val:
            return
        try: 
            if val <= 0:
                raise ValueError("Sampling rate must be positive")
        except:
            raise TypeError("Sampling rate must be a scalar!")
        
        # if it is the first time that a sampling rate is set, do not 
        # modify anything except for self._fs
        if self._fs is None:
            pass
        else:
            warnings.warn(
                "Sampling frequency has been updated! This will modify time "
                "but I'll assume you know what you're doing!"
            )
            self._time = self._tdata / val
        self._fs = val

    @property
    def fs_acquisition(self):
        """(float) Acquisition sampling rate"""
        if self._fs_acquisition is None:
            warnings.warn("No sampling rate specified")
        return self._fs_acquisition

    @fs_acquisition.setter
    def fs_acquisition(self, val):
        try:
            if val > 0:
                self._fs_acquisition = val
                if(self._fs is not  None):
                    self._time = self.tdata / val 
            else:
                raise ValueError("fs_acquisition must be positive")
        except TypeError:
            raise TypeError("fs_acquisition expected to be a scalar")

    @property
    def state(self):
        """(np.array N-D) States associated with timestamps passed in
        """
        if self._state is None:
            warnings.warn("state variable is None")
        return self._state

    @property
    def support(self):
        """(nelpy.EpochArray) The support of the underlying EventArray
        (in seconds).
        """
        return self._support

    @property
    def labels(self):
        """(np.array 1D) Returns N labels 1 for each array of events. 
        """
        return self._labels

    def add_events(self, data, label=None):
        """Docstring goes here.
        Basically, we add another tdata.
        Notes:
        - tdata passed in must be same as what was initiall passed in.
        - single array-like objects or EventArrays are accepted here
        """
        if isinstance(data, EventArray):
            data = data.tdata
        data = np.squeeze(data)

        if data.ndmin > 1:
            raise TypeError("Can only add 1 signal at a time!")
        if self._tdata.ndmin == 1:
            self._tdata = np.vstack([np.array(self._tdata, ndmin=2), \
                                    np.array(data, ndmin=2)])
            if fs is not None:
                sampling_rate = self.fs_acquisition
            else:
                sampling_rate = 1
            self._time = np.vstack([np.array(self._time, ndmin=2), \
                                    np.array(data/sampling_rate, ndmin=2)])
        else:
            self._tdata = np.vstack([self._tdata, np.array(data, ndmin=2)])
            if fs is not None:
                sampling_rate = self.fs_acquisition
            else:
                sampling_rate = 1
            self._time = np.vstack([self._time, \
                                    np.array(data/sampling_rate, ndmin=2)])

    def flatten(self):
        """Docstring goes here.
        """
        raise NotImplementedError("flatten is pretty non-existent atm but"
                                    " it will in the future make all the"
                                    " tdata/time into a 1D array")
        