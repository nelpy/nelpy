# seqtools.py
# helper functions to process sequential data

#import pandas as pd
import numpy as np
#import seaborn as sns

from hmmlearn import hmm # see https://github.com/ckemere/hmmlearn

#from matplotlib import pyplot as plt
#from scipy.signal import butter, lfilter, filtfilt

from mymap import Map

# NOTE: if a sequence is stationary (no movement) then corrXYprob evaluates to nan...
def mXprob(X,prob):
    den = prob.sum()
    num = (prob*X).sum()
    return num/den


def covXYprob(X,Y,prob):
    den = prob.sum()
    num = (prob*(X - mXprob(X,prob))*(Y - mXprob(Y,prob))).sum()
    return num/den


def corrXYprob(X,Y,prob):
    # ignore empty time bins:
    X = X[~np.isnan(prob)]; Y = Y[~np.isnan(prob)]; prob = prob[~np.isnan(prob)]
    den = np.sqrt(covXYprob(X,X,prob)*covXYprob(Y,Y,prob))
    num = covXYprob(X,Y,prob)
    return num/den


def max_jump_dist(P):
    return np.diff(P[~np.isnan(P)]).max()


def test():
    print('this is a test, too!')


def data_stack(mapdata, verbose = False):
    """Take list (num seqs) of list (num obs bins) of ndarrays (num cells x 1) 
    of spk counts and returns a (total num obs) x (num cells) ndarray along 
    with an ndarray containing sequence lengths, in order.
    """
    
    tmp = np.array(mapdata.data)
    
    if len(tmp.shape)==1:
        num_sequences = tmp.shape[0]
        numCells = np.array(tmp[0]).shape[1]
    else:
        print('Only one sequence in data; so data is already in stacked format...')
        num_sequences = 1
        numCells = np.array(tmp[0]).shape[0]
        seq = Map()
        seq['data'] = mapdata.data
        seq['sequence_lengths'] = np.array([tmp.shape[0]])
        seq['bin_width'] = mapdata.bin_width
        seq['boundaries'] = mapdata.boundaries
        seq['boundaries_fs'] = mapdata.boundaries_fs
        return seq

    if verbose:
        print('{} sequences being stacked...'.format(num_sequences))

    SequenceLengths = np.zeros(num_sequences, dtype=np.int)

    for ss in np.arange(0,num_sequences):
        SequenceLengths[ss] = len(tmp[ss])

    
    TotalSequenceLength = np.array(SequenceLengths).sum()

    if verbose:
        print('Total sequence length: {} bins, each of width {} seconds'.format(TotalSequenceLength, mapdata.bin_width))

    StackedData = np.zeros((TotalSequenceLength,numCells), dtype=np.int)
    rr = 0;
    for ss in np.arange(0,num_sequences):
        StackedData[rr:rr+SequenceLengths[ss],:] = np.array(tmp[ss]).astype(int)
        rr = rr+SequenceLengths[ss]
    
    seq = Map()
    seq['data'] = StackedData
    seq['sequence_lengths'] = SequenceLengths
    seq['bin_width'] = mapdata.bin_width
    seq['boundaries'] = mapdata.boundaries
    seq['boundaries_fs'] = mapdata.boundaries_fs

    if verbose:
        if mapdata.bin_width:
            print('Successfully stacked {0} sequences for a total of {1:.2f} seconds of data.'.format(num_sequences,TotalSequenceLength*mapdata.bin_width))

    return seq

def data_split(mapdata, tr=0.3, vl=0.3, ts=0.4, randomseed = None, verbose = False):
    """Split mapdata into train, val, and test sets.

        mapdata is a Map() data object, either stacked or not.
    """

    if randomseed is not None:
        np.random.seed(randomseed)
    
    # normalize tr, vl, and ts proportions:
    tmpsum = tr + vl + ts
    tr = tr / tmpsum
    vl = vl / tmpsum
    ts = ts / tmpsum

    trmap = Map()
    vlmap = Map()
    tsmap = Map()

    trmap.bin_width = mapdata.bin_width
    vlmap.bin_width = mapdata.bin_width
    tsmap.bin_width = mapdata.bin_width
    trmap.boundaries_fs = mapdata.boundaries_fs
    vlmap.boundaries_fs = mapdata.boundaries_fs
    tsmap.boundaries_fs = mapdata.boundaries_fs

    num_sequences = np.array(mapdata.sequence_lengths).shape[0]
    if verbose:
        print('Splitting {} sequences into train, validation, and test sets...'.format(num_sequences))

    indices = np.random.permutation(num_sequences)
    
    tridx = indices[np.arange(0, np.floor(tr*num_sequences)).astype(int)]
    vlidx = indices[np.arange(np.floor(tr*num_sequences), np.floor(tr*num_sequences) + np.floor(vl*num_sequences)).astype(int)]
    tsidx = indices[np.arange(np.floor(tr*num_sequences) + np.floor(vl*num_sequences), num_sequences).astype(int)]
    
    if isinstance(mapdata.data,np.ndarray):
        lstdata = data_lists(mapdata) 
        tmp = np.array(lstdata.data)
    else:   
        tmp = np.array(mapdata.data)
    
    trmap.data = list(tmp[tridx])
    vlmap.data = list(tmp[vlidx])
    tsmap.data = list(tmp[tsidx])

    trmap.boundaries = mapdata.boundaries[tridx]
    vlmap.boundaries = mapdata.boundaries[vlidx]
    tsmap.boundaries = mapdata.boundaries[tsidx]

    trmap.tridx = tridx
    vlmap.vlidx = vlidx
    tsmap.tsidx = tsidx

    if isinstance(mapdata.data,np.ndarray):
        # stack data if the original data was stacked:
        trtmp = data_stack(trmap, verbose = verbose)
        trmap.data = trtmp.data
        trmap.sequence_lengths = trtmp.sequence_lengths
        vltmp = data_stack(vlmap, verbose = verbose)
        vlmap.data = vltmp.data
        vlmap.sequence_lengths = vltmp.sequence_lengths
        tstmp = data_stack(tsmap, verbose = verbose)
        tsmap.data = tstmp.data
        tsmap.sequence_lengths = tstmp.sequence_lengths
        if verbose:
            print('Stacked data split into train ({:.1f} %), validation ({:.1f} %) and test ({:.1f} %) sequences.'.format(tr*100,vl*100,ts*100))
    else:
        if verbose:
            print('List data split into train ({:.1f} %), validation ({:.1f} %) and test ({:.1f} %) sequences.'.format(tr*100,vl*100,ts*100))

    return trmap, vlmap, tsmap

def data_lists(mapdata, verbose = False):
    """convert stacked data into list of sequences

    """

    if mapdata.sequence_lengths is not None:
        if mapdata.sequence_lengths.sum() != mapdata.data.shape[0]:
            print('Data object does not appear to be in a valid stacked format!')
            return None
    else:
        print('Error! Sequences do not appear to be in a valid stacked format!')
        return None

    listmap = Map()
    listmap.bin_width = mapdata.bin_width
    listmap.boundaries = mapdata.boundaries
    listmap.boundaries_fs = mapdata.boundaries_fs
    listmap.sequence_lengths = mapdata.sequence_lengths
    listmap.data = []

    ii = 0
    for ss in mapdata.sequence_lengths:
        listmap.data.append(mapdata.data[ii:ii+ss])
        ii = ii+ss

    return listmap


def hmm_train(trx, num_states=15, n_iter=20, init_params='stm', params='stm', verbose=False):

    if trx.sequence_lengths is not None:
        if trx.sequence_lengths.sum() != trx.data.shape[0]:
            print('Error! Sequences do not appear to be in a valid stacked format!')
            return None
    else:
        print('Error! Sequences do not appear to be in a valid stacked format!')
        return None

    myhmm = hmm.PoissonHMM(n_components=num_states, n_iter=n_iter, init_params=init_params, params=params, verbose=verbose)
    myhmm.fit(trx.data, lengths=trx.sequence_lengths)
    
    return myhmm


def hmm_eval(hmm, obs, symbol_by_symbol=False, verbose=False):
    '''
    evaluates the log probability (!!! really the log likelihood, see e.g. https://github.com/hmmlearn/hmmlearn/issues/20) 
    of a sequence of observations, marginalizing out all the possible hidden states. (Forward alg)

    returns a generator

    Compute the log probability under the model.

    Parameters :

    obs : array_like, shape (n, n_cells) : Sequence of n_cells-dimensional data points. Each row corresponds to a single data point.
    Returns :

    logprob : float : Log likelihood of the obs, as a generator.
    '''
    # http://stats.stackexchange.com/questions/79955/scikit-learn-gaussianhmm-decode-vs-score
    if verbose:
        print('building generator containing log likelihoods of observation sequences for every sequence...')
    if isinstance(obs, Map):

        if obs.sequence_lengths is not None:
            if obs.sequence_lengths.sum() != obs.data.shape[0]:
                if verbose:
                    print('Sequences are not stacked!')
                return (hmm.score(seq) for seq in obs.data)
            else:
                if verbose:
                    print('Sequences are stacked!')
                assert obs.sequence_lengths[0] > 0, "observation sequences do not have expected sequence lengths"
                obs_lists = data_lists(obs)
                return (hmm.score(seq) for seq in obs_lists.data)
        else:
            if verbose:
                print('Sequences are not stacked!')
            return (hmm.score(seq) for seq in obs.data)

    elif isinstance(obs, np.ndarray):
        if verbose:
            print('Single sequence in ndarray!')
        if symbol_by_symbol:  # evaluate
            return (hmm.score(obst.reshape(1,-1)) for obst in obs)
        else:
            return iter([hmm.score(obs)])
    elif isinstance(obs, list):
        if verbose:
            print('Sequences are not stacked!')
        return (hmm.score(seq) for seq in obs)
    else:
        print(type(obs))
    return None


def hmm_decode(hmm, obs, algorithm='viterbi', verbose=False):
    '''
    evaluates the log probability (!!! really the log likelihood, see e.g. https://github.com/hmmlearn/hmmlearn/issues/20) 
    of a sequence of observations, marginalizing out all the possible hidden states. (Forward alg)

    returns a tuple of generators: lp, pth = hmm_decode(...)
    '''
    # http://stats.stackexchange.com/questions/79955/scikit-learn-gaussianhmm-decode-vs-score
    if verbose:
        print('building generator containing log likelihoods of observation sequences for every sequence...')
    if isinstance(obs, Map):

        if obs.sequence_lengths is not None:
            if obs.sequence_lengths.sum() != obs.data.shape[0]:
                print('Sequences are not stacked!')
                return (hmm.decode(seq, algorithm=algorithm)[0] for seq in obs.data), (hmm.decode(seq, algorithm=algorithm)[1] for seq in obs.data)
            else:
                print('Sequences are stacked!')
                assert obs.sequence_lengths[0] > 0, "observation sequences do not have expected sequence lengths"
                obs_lists = data_lists(obs)
                return (hmm.decode(seq, algorithm=algorithm)[0] for seq in obs_lists.data), (hmm.decode(seq, algorithm=algorithm)[1] for seq in obs_lists.data)
        else:
            print('Sequences are not stacked!')
            return (hmm.decode(seq, algorithm=algorithm)[0] for seq in obs.data), (hmm.decode(seq, algorithm=algorithm)[1] for seq in obs.data)

    elif isinstance(obs, np.ndarray):
        return iter([hmm.decode(obs, algorithm=algorithm)[0]]), iter([hmm.decode(obs, algorithm=algorithm)[1]])
    elif isinstance(obs, list):
        print('Sequences are not stacked!')
        return (hmm.decode(seq, algorithm=algorithm)[0] for seq in obs), (hmm.decode(seq, algorithm=algorithm)[1] for seq in obs)
    else:
        print(type(obs))
    return None
