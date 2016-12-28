# klabtools.py
# helper functions to process ephys data

# import os.path
# import sys
# import scipy.io
import pandas as pd
import numpy as np
# import re
import seaborn as sns

from matplotlib import pyplot as plt
from scipy.signal import butter, lfilter, filtfilt

import ripple
from mymap import Map


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filtfilt(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data, padlen=150)
    return y


def plot_trajectory(posdf=0,x1=0,x2=0,y1=0,y2=0, vs_time=False, fs=60):
    if isinstance(posdf, pd.DataFrame):
        if vs_time:
            time_axis = np.linspace(0, len(posdf.index)/fs, len(posdf.index))
            plt.plot(time_axis, (posdf['x1']+posdf['x2'])/2, color='k')
            plt.plot(time_axis, (posdf['y1']+posdf['y2'])/2, color='lightgray')
            plt.xlabel('time (s)')
            plt.ylabel('position')
            plt.legend(['x', 'y'])
        else:
            plt.plot(posdf['x1'], posdf['y1'], color='k')
            plt.plot(posdf['x2'], posdf['y2'], color='lightgray')
            plt.plot((posdf['x1'] + posdf['x2'])/2, (posdf['y1'] + posdf['y2'])/2, linewidth=1, color='k')
            plt.xlabel('x (units)')
            plt.ylabel('y (units)')
    else:
        if vs_time:
            time_axis = np.linspace(0, len(x1)/fs, len(x1))
            plt.plot(time_axis, (x1+x2)/2, linewidth=2, color='k')
            plt.plot(time_axis, (y1+y2)/2, linewidth=2, color='lightgray')
            plt.xlabel('time (s)')
            plt.ylabel('position')
            plt.legend(['x', 'y'])
        else:
            plt.plot(x1, y1, linewidth=2, color='lightgray')
            plt.plot(x2, y2, linewidth=2, color='lightgray')
            plt.plot((x1 + x2)/2, (y1 + y2)/2, linewidth=1, color='k' )
            plt.xlabel('x (units)')
            plt.ylabel('y (units)')

def get_smooth_speed(posdf,fs=60,th=3,cutoff=0.5,showfig=False,verbose=False):
    x = (np.array(posdf['x1']) + np.array(posdf['x2']))/2
    y = (np.array(posdf['y1']) + np.array(posdf['y2']))/2

    dx = np.ediff1d(x,to_begin=0)
    dy = np.ediff1d(y,to_begin=0)
    dvdt = np.sqrt(np.square(dx) + np.square(dy))*fs # units (cm?) per second
    t0 = 0
    tend = len(dvdt)/fs # end in seconds

    dvdtlowpass = np.fmax(0,butter_lowpass_filtfilt(dvdt, cutoff=cutoff, fs=fs, order=6))

    if verbose:
        print('The animal (gor01) ran an average of {0:2.2f} units/s'.format(dvdt.mean()))

    #th = 3 #cm/s
    
    runindex = np.where(dvdtlowpass>=th); runindex = runindex[0]
    if verbose:
        print("The animal ran faster than th = {0:2.1f} units/s for a total of {1:2.1f} seconds (out of a total of {2:2.1f} seconds).".format(th,len(runindex)/fs,len(x)/fs))
    
    if showfig:
        #sns.set(rc={'figure.figsize': (15, 4),'lines.linewidth': 3, 'font.size': 18, 'axes.labelsize': 16, 'legend.fontsize': 12, 'ytick.labelsize': 12, 'xtick.labelsize': 12 })
        #sns.set_style("white")

        f, (ax1, ax2) = plt.subplots(1,2)

        ax1.plot(np.arange(0,len(dvdt))/fs,dvdt,alpha=1,color='lightgray',linewidth=2)
        ax1.plot(np.arange(0,len(dvdt))/fs,dvdtlowpass, alpha=1,color='k',linewidth=1)
        ax1.set_xlabel('time (seconds)')
        ax1.set_ylabel('instantaneous velocity (units/s)')
        ax1.legend(['unfiltered', str(cutoff) + ' Hz lowpass filtfilt'])
        ax1.set_xlim([0,10*np.ceil(len(x)/fs/10)])

        ax2.plot(np.arange(0,len(dvdt))/fs,dvdt,alpha=1,color='lightgray',linewidth=2)
        ax2.plot(np.arange(0,len(dvdt))/fs,dvdtlowpass, alpha=1,color='k',linewidth=1)
        ax2.set_xlabel('time (seconds)')
        ax2.set_ylabel('instantaneous velocity (units/s)')
        ax2.legend(['unfiltered',  str(cutoff) + ' Hz lowpass filtfilt'])
        ax2.set_xlim([30,70])

    speed = Map()
    speed['data'] = dvdtlowpass
    speed['active_bins'] = runindex
    speed['active_thresh'] = th
    speed['samprate'] = fs
  
    return speed

def bin_spikes(st_array_extern, ds=0, fs=0, boundaries=None, boundaries_fs=1252, verbose=False ):
    """
    st_array: list of ndarrays containing spike times (in sample numbers! Currently assumes first sample number is 0)
    ds:       bin width in seconds
    fs:       sampling frequency of spikes
    
    returns a (numBins x numCell) array with spike counts
    """
    st_array = st_array_extern
    
    # determine number of units:
    num_units = len(st_array)
    
    binned = Map()
    binned['bin_width'] = ds

    if boundaries is None:
        num_bins = int(np.ceil(np.max([np.max(x) for x in st_array if len(x)>0])/fs/ds))
        maxtime = num_bins*ds

        spks_bin = np.zeros((num_bins,num_units))

        if verbose:
            print("binning approx {0} s of data into {1} x {2:2.1f} ms temporal bins...".format(maxtime, num_bins, ds*1000))

        for uu in np.arange(num_units):
            # count number of spikes in an interval:
            spks_bin[:,uu] = np.histogram(st_array[uu]/fs, bins=num_bins, density=False, range=(0,maxtime))[0]

        binned['data'] = spks_bin
    else:
        #TODO: rewrite enire 'else' section to make it faster and more pythonic
        st_array = [x*(boundaries_fs/fs) for x in st_array_extern] # spike times in fsEEG sample numbers (not integral)

        from math import ceil       
        num_events = boundaries.shape[0] # number of events
        spk_cnters = np.zeros((num_units,1),dtype=np.int)

        EventSpikes = []
        idx = -1
        for ee in np.arange(0,num_events):
            start = boundaries[ee,0]
            stop = boundaries[ee,1]
            duration = (stop - start)/boundaries_fs
            num_bins = ceil(duration/ds)
            EventSpikes.append([]) # list for idx=nth SWR event
            for bb in np.arange(0,num_bins):
                EventSpikes[ee].append([]) # add list element for each bin in sequence
                for uu in np.arange(0,num_units):
                    # count spikes in bin and advance spike time array counter to make subsequent searches faster:
                    spk_cnters[uu][0] = spk_cnters[uu][0] + len(st_array[uu][spk_cnters[uu][0]:][st_array[uu][spk_cnters[uu][0]:]<start+(bb)*ds*boundaries_fs])
                    #debug#print("skip first {0} spikes for unit {1}...".format(spk_cnters[uu][0],uu))
                    tempspikes = st_array[uu][spk_cnters[uu][0]:][st_array[uu][spk_cnters[uu][0]:]<=start+(bb+1)*ds*boundaries_fs]
                    numspikes = len(tempspikes)
                    #print("spikes in bin {0} of unit {1}: {2}".format(bb,uu,numspikes))
                    EventSpikes[idx][bb].append(np.array(numspikes))

        binned['data'] = EventSpikes
        binned['boundaries'] = boundaries
        binned['boundaries_fs'] = boundaries_fs

    return binned

def estimate_place_fields(lin_pos,binned_spk_cnts,fs,x0,xl,num_pos_bins=200,minth=0.005,max_meanfiringrate=4,min_maxfiringrate=3,sigma=1, verbose=False, showfig=False):
    # estimate place fields using ALL the spikes in lin_pos and binned_spk_cnts... therefore, as inputs, you should pass lin_pos and 
    # binned_spk_times only for active periods (bins)... this is made simple with runidx which is returned by get_smooth_speed()
    # maybe I should incorporate all of this in the estimate_place_fields function? That is, pass a movement threshold and do the
    # rest internally?
    
    from scipy.ndimage.filters import gaussian_filter1d

    
    num_units = len(binned_spk_cnts[0])

    bins_left = np.linspace(x0,xl,num_pos_bins+1)
    pfbincenters = bins_left[:-1] + np.diff(bins_left)/2
    digitized = np.digitize(lin_pos, bins_left) - 1 # bin numbers

    bin_cnt = [len(lin_pos[digitized == i]) for i in range(0, len(bins_left))]
    bin_time = [b/fs for b in bin_cnt] # convert to seconds spent in bin

    pf2spk_cnt = np.zeros((num_pos_bins,num_units))

    for cnt, bb in enumerate(digitized):
        pf2spk_cnt[bb,:] += binned_spk_cnts[cnt,:]
           
    if showfig:
        fig, ax1 = plt.subplots()

        ax1.plot(pf2spk_cnt, 'gray', linewidth=1)
        #DEBUG: plot only cells 11 and 20 for debugging...
        #ax1.plot(pf2spk_cnt[:,[11,20]], 'gray', linewidth=1)
        ax1.set_xlabel('position (cm)')
        # Make the y-axis label and tick labels match the line color.
        ax1.set_ylabel('spikes per bin', color='gray')
        for tl in ax1.get_yticklabels():
            tl.set_color('gray')

        ax2 = ax1.twinx()
        ax2.plot(bin_time,'DarkCyan',linewidth=2,marker='o')
        ax2.set_ylabel('time spent in position bin (sec)', color='DarkCyan')
        for tl in ax2.get_yticklabels():
            tl.set_color('DarkCyan')
    
    pf2 = []
    pfsmooth = []
    
#    minth = 0.05 # min threshold for backgrnd spking activity
    for uu in np.arange(0,num_units):
        pf2.append([b/max(c,1/fs) for (b,c) in zip(pf2spk_cnt[:,uu],bin_time)])
        pfsmooth.append(gaussian_filter1d(pf2[uu], sigma=sigma))

    pfsmooth = np.array(pfsmooth)
    pfsmooth[pfsmooth<minth] = minth # enforce a minimum background firing rate.
    
    # throw away cells that look like interneurons, or cells that are inactive throughout the entire experiment:
    meanfiringrates = pfsmooth.mean(axis=1)
    maxfiringrates = pfsmooth.max(axis=1)

    pindex = np.where((meanfiringrates<=max_meanfiringrate) & (maxfiringrates>min_maxfiringrate)); pindex = pindex[0]
    if verbose:
        print("{0} out of {1} cells passed the criteria to be place cells...".format(len(pindex),len(meanfiringrates)))

    return pfsmooth, pfbincenters, pindex


def show_place_fields(pfs, pfbincenters, pindex,min_maxfiringrate=0):
    if len(pindex)==0:
        print('No place cells!')
        return

    meanfiringrates = pfs.mean(axis=1)
    maxfiringrates = pfs.max(axis=1)

    # visualize place fields

    # order remaining cells by peak hight along the track
    peaklocations = pfs.argmax(axis=1)
    peakorder = peaklocations[pindex].argsort()

    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    ax1.plot(meanfiringrates, linewidth=2, color='lightgray')
    ax1.plot(maxfiringrates, linewidth=1, color='k')
    ax1.legend(['mean firing rate', 'max firing rate'])
    ax1.set_title('mean and max firing rates of all cells')

    ax2.plot(meanfiringrates[pindex],linewidth=2,color='lightgray')
    ax2.plot(maxfiringrates[pindex],linewidth=1,color='k')
    ax2.legend(['mean firing rate','max firing rate'])
    ax2.set_title('mean and max firing rates of place cells')

    cell_list = pindex

    for uu in cell_list:
        ax3.plot(pfbincenters, pfs[uu], linewidth=1, color='k')

    plt.subplots_adjust(hspace=0.40)
    #ax3.set_title("place fields in LinearTwo",fontsize=14)
    ax1.set_ylabel("firing rate (Hz)")
    ax2.set_ylabel("firing rate (Hz)")
    ax3.set_ylabel("firing rate (Hz)")
    ax3.set_xlim([0,100])
    ax3.set_title('Place fields of place cells')
    
    sns.set(rc={'figure.figsize': (4,8),'lines.linewidth': 3, 'font.size': 18, 'axes.labelsize': 16, 'legend.fontsize': 12, 'ytick.labelsize': 12, 'xtick.labelsize': 12 })
    sns.set_style("white")
    
    f, axes = plt.subplots(len(pindex), sharex=True, sharey=True)
    plt.subplots_adjust(hspace=0.100)

    for ii,pp in enumerate(pindex[peakorder]):
        axes[ii].plot((0, 100), (min_maxfiringrate, min_maxfiringrate), 'k:', linewidth=1)
        axes[ii].plot(pfbincenters, pfs[pp],linewidth=1,color='k')
        axes[ii].fill_between(pfbincenters, 0, pfs[pp], color='DarkCyan',alpha=0.25)
        axes[ii].set_xticks([])
        axes[ii].set_yticks([])
        axes[ii].spines['top'].set_visible(False)
        axes[ii].spines['right'].set_visible(False)
        axes[ii].spines['bottom'].set_visible(False)
        axes[ii].spines['left'].set_visible(False)
        axes[ii].set_ylabel(pp, fontsize=12)
        axes[ii].set_ylim([0,15])

    axes[-1].set_xticks([10,50,90])
    axes[-1].set_xlabel('position along track [cm]')
    f.suptitle('Place fields ordered by peak location along track, cells are zero-indexed.')

    sns.set(rc={'figure.figsize': (12, 4),'lines.linewidth': 1, 'font.size': 18, 'axes.labelsize': 16,
            'legend.fontsize': 12, 'ytick.labelsize': 12, 'xtick.labelsize': 12 })
    sns.set_style("white")


def GenerateSpikes(IntensityFunc, MaxRate, PositionFunc, TotalTime) :
    # Start by generating spikes for a homogeneous Poisson process
    nHomogeneousSpikes = np.random.poisson(MaxRate * TotalTime);
    tHomogeneousSpikeTimes = np.random.uniform(0,TotalTime,nHomogeneousSpikes);
    tHomogeneousSpikeTimes.sort();

    if (tHomogeneousSpikeTimes.size > 0) :
      # Next, we need to evaluate intensity function at the locations/times of
      #  our generated spikes. inhomogeneousRate is an ndarray vector of the
      #  same length as the spike times.
      inhomogeneousRate = IntensityFunc( PositionFunc(tHomogeneousSpikeTimes),
                                          tHomogeneousSpikeTimes);
      
      # The we'll compare the ratio of the inhomogeneousRates and the MaximumRate
      #  to a random number generator to decide when/when-not to delete.
      rnd = np.random.uniform(0, 1, inhomogeneousRate.size);
      tSpikeTimes = tHomogeneousSpikeTimes[inhomogeneousRate/MaxRate > rnd];
      return tSpikeTimes;
    else :
      return tHomogeneousSpikeTimes;


def generate_spikes_from_traj(binned_runidx,truepos,pfs,pfbincenters,pos_fs,const_firing_rate=False,verbose=False):

    # extract running trajectories from real data:
    run_ends = np.where(np.diff(binned_runidx)-1)[0] + 1
    seq_lengths = np.diff(np.hstack((0, run_ends, binned_runidx.size)))
    runbdries = np.hstack((0, run_ends))

    # each spike train corresponding to a trajectory is generated between [0,tend)
    # and then a time offset is added, so that the overall spike times are consistent
    # with that of the original experiment. In that way, when we re-bin and
    # select those bins with run_vel > th, we will get the correct spikes.

    # seq_lengths[sseq_lengths<bins_per_window]

    fs = 32552
    numCells = pfs.shape[0]
    NumTrajectories = len(runbdries)
    SpikeRasters = [[[] for _ in range(numCells)] for _ in range(NumTrajectories)] 

    tempsum = 0
    tempdur = 0

    for ii in np.arange(len(runbdries)-1):
        for nn in np.arange(numCells):
            traj_start_bin = binned_runidx[runbdries[ii]]
            traj_end_bin = binned_runidx[runbdries[ii+1]-1] # inclusive
            #print("Trajectory {0} length is {1} bins, or {2:2.2f} seconds.".format(ii,traj_end_bin-traj_start_bin + 1, (traj_end_bin-traj_start_bin + 1)/pos_fs))
            t_offset = traj_start_bin/pos_fs # in seconds
            posvect = truepos[runbdries[ii]:runbdries[ii+1]]
            TrajDuration = len(posvect)/pos_fs
            #print("Traj duration {0:2.2f}".format(TrajDuration))
            tvect = np.linspace(0,TrajDuration, len(posvect)) # WARNING! This does not work when traj_len == 1, but this should never happen anyway

            posFun = lambda t: np.interp(t,tvect,posvect)
            if const_firing_rate:
                PlaceFieldRate = lambda x: 0*x + 10 #np.interp(x,pfbincenters,pfs[nn,:])
            else:
                PlaceFieldRate = lambda x: np.interp(x,pfbincenters,pfs[nn,:])
            maxrate = pfs[nn,:].max()+25

            SpikeRasters[ii][nn] = np.round((t_offset + GenerateSpikes(lambda x,t : PlaceFieldRate(x), maxrate, posFun, TrajDuration))*fs);
            
             # DEBUG:
            tempsum += len(SpikeRasters[ii][nn])
            tempdur += TrajDuration
    if verbose:
        print('true average spike rate synthesized (across all cells), before place field estimation: {0:1.2f} Hz'.format(tempsum/tempdur))

    # consolidate spike rasters (not stratified by trajectory)
    cons_st_array = []
    for nn in np.arange(numCells):
        st = np.zeros(0)
        for ii in np.arange(NumTrajectories):
            st = np.hstack((st, SpikeRasters[ii][nn]))
        st = np.sort(st)
        cons_st_array.append(st)

    return cons_st_array


def detect_ripples(data, FS=1252, ThresholdSigma=3, SecondaryThresholdSigma=0, LengthCriteria=0.015):
    # wrapper function for Caleb's ripple.py ripple detection
    ripple_bounds, ripple_maxes, ripple_events, ripple_data, ripple_envelope, smoothed_envelope = ripple.detect(data, FS=FS, ThresholdSigma=ThresholdSigma, SecondaryThresholdSigma=SecondaryThresholdSigma, LengthCriteria=LengthCriteria)

    ripples = Map()

    ripples['samprate'] = FS
    ripples['ripple_bounds'] = ripple_bounds
    ripples['ripple_maxes'] = ripple_maxes
    ripples['ripple_events'] = ripple_events
    ripples['ripple_data'] = ripple_data
    ripples['ripple_envelope'] = ripple_envelope
    ripples['smoothed_envelope'] = smoothed_envelope
    ripples['criteria'] = ['ThresholdSigma=' + str(ThresholdSigma), 'LengthCriteria=' + str(LengthCriteria)]

    return ripples


# note that here we re-sample placefields simply with linear interpolation. A more 'accurate' approach might be to compute the mean firing rate within each new bin...
def resample_placefields(pfsmooth, s_bin, pfbincenters, x0,xl):
    # pfsmooth is NCells x bins containing discretized firing rates
    # s_bin is desired spatial bin size
    # x0 and xl are starting and ending points for LINEARIZED segment to re-sample
    # returns an NCells x new_bins, and newx, the bin center positions
    newx = np.arange(x0,xl,s_bin) + s_bin/2
    ss_pfsmooth = np.zeros((pfsmooth.shape[0],len(newx)))
    for cc in np.arange(0,pfsmooth.shape[0]):
        ss_pfsmooth[cc,:] = np.interp(newx,pfbincenters,pfsmooth[cc,:])
    return ss_pfsmooth, newx

def resample_velocity(velocity, t_bin, tvel, t0,tend):
    # velocity is a vector (ndarray) of velocities
    # tvel is the same length as velocity, and contains the times (or sample #s) associated with those velocities
    # t_bin is the desired time bin for the resampled velocities, with start and end times t0 and tend, respectively
    newt = np.arange(t0,tend,t_bin) + t_bin/2
    newvel = np.zeros((1,len(newt)))
    newvel = np.interp(newt,tvel,velocity)
    return newvel, newt


def decode_swr_sequence(seq,pfs,pfbincenters,pindex,t_bin,tau,showfig=False,shuffletype='none'):
    
    # decode a sequence of observations using 20 ms window with 5 ms sliding window increments
    
    #n  C x 1        changes every time step
    #fi C x 1        never changes
    #f  C x nbins    never changes

    seq = np.array(seq)
    bins_per_window = round(tau/t_bin)
    num_tbins=len(seq)
    PP = np.zeros((len(pfbincenters),num_tbins-bins_per_window))
    f = pfs[pindex,:]
    if shuffletype == 'unit-all':
        np.random.shuffle(pindex) # unit identity shuffle using ALL place cells
    elif shuffletype == 'unit-event':
        # determine which place cells participate in the event:
        eventpcells = pindex[np.nonzero(seq.sum(axis=0)[pindex])]
        # unit identity shuffle using only participating place cells
        np.random.shuffle(eventpcells)
        pindex[np.nonzero(seq.sum(axis=0)[pindex])] = eventpcells
    dec_pos = np.zeros((num_tbins-bins_per_window,1))
    prob = np.zeros((num_tbins-bins_per_window,1))
    est_pos_idx = 0
        
    for tt in np.arange(0,num_tbins-bins_per_window): #len(spk_counts2_5ms_run)-4):
        #tt+=1 # time index
        n = seq[tt:tt+bins_per_window,pindex].sum(axis=0)
        nn = np.tile(n,(len(ss_pfbincenters),1)).T
        if nn.max() == 0:
            #print('No spikes in decoding window, so cannot decode position!')
            PP[:,tt] = PP[:,tt]
            est_pos_idx = np.nan
            dec_pos[tt] = np.nan
            prob[tt] = np.nan
        else:
            # print('Some spikes detected in decoding window.. yeah!!!')
            PP[:,tt] = np.exp((np.log((f)**(nn))).sum(axis=0) - tau*f.sum(axis=0))
            PP[:,tt] = PP[:,tt]/PP[:,tt].sum() # normalization not strictly necessary
            est_pos_idx = PP[:,tt].argmax()
            dec_pos[tt] = ss_pfbincenters[est_pos_idx]
            prob[tt] = PP[est_pos_idx,tt]
    T = np.arange(0,num_tbins-bins_per_window)*t_bin*1000
    T = T.reshape((len(T),1))
    if showfig:
        sns.set(rc={'figure.figsize': (16, 6),'lines.linewidth': 3, 'font.size': 16, 'axes.labelsize': 14, 'legend.fontsize': 12, 'ytick.labelsize': 12, 'xtick.labelsize': 12 })
        sns.set_style("white")
        f, (ax1, ax2) = plt.subplots(1,2)

        x0=0
        xl=np.ceil(pfbincenters[-1]/10)*10
        tend=num_tbins*5
        extent=(0,T[-1],x0,xl)
        ax1.imshow(PP,cmap='PuBu',origin='lower',extent=extent,interpolation='none')
        yticks=np.arange(x0,xl+1,20)
        ax1.set_yticks(yticks)
        ax1.set_ylabel('position (cm)')
        ax1.set_xlabel('time (ms)')
        ax2.plot(T, dec_pos,marker='o')
        ax2.set_aspect('equal')
        ax2.set_ylim([x0,xl])
        ax2.set_xlim([0,T[-1]])
        ax2.set_yticks(yticks)
        ax2.set_ylabel('position (cm)')
        ax2.set_xlabel('time (ms)')
    
    return T, dec_pos, prob, PP

def get_continuous_segments(bins):
    from itertools import groupby
    from operator import itemgetter

    data = np.sort(bins)
    bdries = []

    for k, g in groupby(enumerate(data), lambda ix: ix[0] - ix[1]):
        f = itemgetter(1)
        gen = (f(x) for x in g)
        start = next(gen)
        stop = start
        for stop in gen:
            pass
        bdries.append([start, stop])
    return np.asarray(bdries)

def get_boundaries_from_bins(fs, bins, bins_fs):
    '''
    bins is an ndarray containing integral, zero-indexed bin numbers for which some condition is true, e.g. min velocity.
    getbdyfrombins returns a (Nepochs x 2) ndarray with each row corresponding to [start; stop] times in fs sample numbers
    '''

    from itertools import groupby
    from operator import itemgetter

    data = np.sort(bins)
    #bdries = []

    #for k, g in groupby(enumerate(data), lambda ix: ix[0] - ix[1]):
    #    f = itemgetter(1)
    #    gen = (f(x) for x in g)
    #    start = next(gen)
    #    stop = start
    #    for stop in gen:
    #        pass
    #    bdries.append([int(np.floor(start/bins_fs*fs)), int(np.ceil(stop/bins_fs*fs))])

    bdries = get_continuous_segments(bins)
    bdries[:,0] = np.floor(bdries[:,0]/bins_fs*fs)
    bdries[:,1] = np.ceil(bdries[:,1]/bins_fs*fs)

    return bdries

def split(dataobj, datatype, changept, verbose=False):

    if (datatype=='spikes'):
        num_units = len(dataobj.data)
        spklist1 = []
        spklist2 = []
        for uu in np.arange(0,num_units):
            tempspikearray = np.asarray(dataobj.data[uu])
            tsa1 = tempspikearray[tempspikearray<=changept]
            tsa2 = tempspikearray[tempspikearray>changept] - changept
            spklist1.append(tsa1)
            spklist2.append(tsa2)

        spikes1 = Map()
        spikes1['data'] = spklist1
        spikes1['num_electrodes'] = dataobj.num_electrodes
        spikes1['num_units'] = num_units
        spikes1['samprate'] = dataobj.samprate
        spikes1['session'] = dataobj.session
        spikes1['offset'] = 0

        spikes2 = Map()
        spikes2['data'] = spklist2
        spikes2['num_electrodes'] = dataobj.num_electrodes
        spikes2['num_units'] = num_units
        spikes2['samprate'] = dataobj.samprate
        spikes2['session'] = dataobj.session
        spikes2['offset'] = changept

        return spikes1, spikes2

    elif (datatype=='eeg'):
        data_arr1 = dataobj.data[:changept, :]
        data_arr2 = dataobj.data[changept:, :]

        eeg1 = Map()
        eeg1['data'] = data_arr1
        eeg1['channels'] = dataobj.channels
        eeg1['samprate'] = dataobj.samprate
        eeg1['starttime'] = dataobj.starttime
        eeg1['session'] = dataobj.session

        eeg2 = Map()
        eeg2['data'] = data_arr2
        eeg2['channels'] = dataobj.channels
        eeg2['samprate'] = dataobj.samprate
        eeg2['starttime'] = dataobj.starttime + changept
        eeg2['session'] = dataobj.session
        
        return eeg1, eeg2

    elif (datatype=='pos'):
        return dataobj[:changept], dataobj[changept:]

    else:
        raise ValueError('datatype is not handled')

        