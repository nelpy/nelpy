# placefieldviz.py
# temp script to vizualise 

# import os.path
# import sys
# import scipy.io
import numpy as np
import klabtools as klab
import seqtools as sq

from matplotlib import pyplot as plt

from mymap import Map


def extract_subsequences_from_binned_spikes(binned_spikes, bins):
    boundaries = klab.get_continuous_segments(bins)
    
    binned = Map()
    binned['bin_width'] = binned_spikes.bin_width
    binned['data'] = binned_spikes.data[bins,:]
    binned['boundaries'] = boundaries
    binned['boundaries_fs'] = 1/binned_spikes.bin_width   
    binned['sequence_lengths'] = (boundaries[:,1] - boundaries[:,0] + 1).flatten()
    
    return binned

def get_sorted_order_from_transmat(A, start_state = 0):
    
    num_states = A.shape[0]
    cs = np.min([start_state, num_states-1])
    new_order = [cs]
    rem_states = np.arange(0,cs).tolist()
    rem_states.extend(np.arange(cs+1,num_states).tolist())
    
    for ii in np.arange(0,num_states-1):
        nstilde = np.argmax(A[cs,rem_states])
        ns = rem_states[nstilde]
        rem_states.remove(ns)
        cs = ns
        new_order.append(cs)
        
    return new_order, A[:, new_order][new_order]

def hmmplacefieldposviz(spikes, speed, posdf, num_states=35, ds=0.0625, vth=8, normalize=False, verbose=False, experiment='both'):
    binned_spikes = klab.bin_spikes(spikes.data, ds=ds, fs=spikes.samprate, verbose=verbose)

    centerx = (np.array(posdf['x1']) + np.array(posdf['x2']))/2
    centery = (np.array(posdf['y1']) + np.array(posdf['y2']))/2

    tend = len(speed.data)/speed.samprate # end in seconds
    time_axis = np.arange(0,len(speed.data))/speed.samprate
    speed_ds, tvel_ds = klab.resample_velocity(velocity=speed.data,t_bin=ds,tvel=time_axis,t0=0,tend=tend)
    truepos_ds = np.interp(np.arange(0,len(binned_spikes.data))*ds,time_axis,centerx)

    # get bins where rat was running faster than thresh units per second
    runidx_ds = np.where(speed_ds>vth)[0]
    # filter based ob 'first', 'second', or 'both' halves of the experiment:
    if experiment=='first':
        runidx_ds = runidx_ds[runidx_ds<=len(speed_ds)/2]
    elif experiment=='second':
        runidx_ds = runidx_ds[runidx_ds>=len(speed_ds)/2]

    seq_stk_run_ds = extract_subsequences_from_binned_spikes(binned_spikes,runidx_ds)

    ## split data into train, test, and validation sets: (WARNING! MAY BREAK ON EMPTY SETS!!)
    tr_b,vl_b,ts_b = sq.data_split(seq_stk_run_ds, tr=50, vl=10, ts=50, randomseed = 0, verbose=verbose)

    ## train HMM on active behavioral data; training set (with a fixed, arbitrary number of states for now):
    myhmm = sq.hmm_train(tr_b, num_states=num_states, n_iter=50, verbose=verbose)

    ###########################################################3
    stacked_data = seq_stk_run_ds
    ###########################################################3

    x0=0; xl=100; num_pos_bins=50
    xx_left = np.linspace(x0,xl,num_pos_bins+1)
    num_sequences = len(stacked_data.sequence_lengths)
    num_states = myhmm.n_components
    state_pos = np.zeros((num_states, num_pos_bins))

    for seq_id in np.arange(0,num_sequences):
        tmpseqbdries = [0]; tmpseqbdries.extend(np.cumsum(stacked_data.sequence_lengths).tolist());
        obs = stacked_data.data[tmpseqbdries[seq_id]:tmpseqbdries[seq_id+1],:]
        ll, pp = myhmm.score_samples(obs)
        xx = truepos_ds[stacked_data.boundaries[seq_id,0]:stacked_data.boundaries[seq_id,1]+1]
        digitized = np.digitize(xx, xx_left) - 1 # spatial bin numbers
        for ii, ppii in enumerate(pp):
            state_pos[:,digitized[ii]] += np.transpose(ppii)
            
    ## now order states by peak location on track
    peaklocations = state_pos.argmax(axis=1)
    peakorder = peaklocations.argsort()

    if normalize:
        state_pos = state_pos/np.transpose(np.tile(state_pos.sum(axis=1),[state_pos.shape[1],1]))

    stateorder, Anew = get_sorted_order_from_transmat(myhmm.transmat_, start_state = 17)

    return state_pos, peakorder, stateorder
