"""Temporary scoring functions. Needs a lot of work. DEPRECATED"""

import numpy as np
from itertools import groupby

def scoreOrderND(hmm, state_sequences):
    """Compute order score with no adjacent duplicates in state sequences

    A score of 0 means there's only one state.
    """

    scoresND = [] # scores with no adjacent duplicates

    for seqid in range(len(state_sequences)):
        logP = np.log(hmm.transmat_)
        pth = [x[0] for x in groupby(state_sequences[seqid])] # remove adjacent duplicates
        plen = len(pth)
        logPseq = 0
        for ii in range(plen-1):
            logPseq += logP[pth[ii],pth[ii+1]]
        score = logPseq - np.log(plen)
        scoresND.append(score)

    return np.array(scoresND)

def scoreOrderD_time_swap(hmm, state_sequences, n_shuffles=250):
    """Compute order score of state sequences

    A score of 0 means there's only one state.
    """

    scoresD = [] # scores with no adjacent duplicates
    n_sequences = len(state_sequences)
    shuffled = np.zeros((n_shuffles, n_sequences))

    for seqid in range(n_sequences):
        logP = np.log(hmm.transmat_)
        pth = state_sequences[seqid]
        plen = len(pth)
        logPseq = 0
        for ii in range(plen-1):
            logPseq += logP[pth[ii],pth[ii+1]]
        score = logPseq - np.log(plen)
        scoresD.append(score)
        for nn in range(n_shuffles):
            logPseq = 0
            pth = np.random.permutation(pth)
            for ii in range(plen-1):
                logPseq += logP[pth[ii],pth[ii+1]]
            score = logPseq - np.log(plen)
            shuffled[nn, seqid] = score

    scoresD = np.array(scoresD)
    return scoresND, shuffled

# def scoreOrderD_trans_mat_shuffle(hmm, state_sequences, n_shuffles=5):
#     """Compute order score of state sequences

#     A score of 0 means there's only one state.
#     """

#     scoresND = [] # scores with no adjacent duplicates
#     shuffles =
#     for seqid in range(len(state_sequences)):
#         logP = np.log(hmm.transmat_)
#         pth = state_sequences[seqid]
#         plen = len(pth)
#         logPseq = 0
#         for ii in range(plen-1):
#             logPseq += logP[pth[ii],pth[ii+1]]
#         score = logPseq - np.log(plen)
#         scoresND.append(score)

#     return np.array(scoresND)

def scoreOrderD(hmm, state_sequences):
    """Compute order score of state sequences

    A score of 0 means there's only one state.
    """

    scoresND = [] # scores with no adjacent duplicates
    for seqid in range(len(state_sequences)):
        logP = np.log(hmm.transmat_)
        pth = state_sequences[seqid]
        plen = len(pth)
        logPseq = 0
        for ii in range(plen-1):
            logPseq += logP[pth[ii],pth[ii+1]]
        score = logPseq - np.log(plen)
        scoresND.append(score)

    return np.array(scoresND)

def scoreOrderNAND(hmm, state_sequences):
    """Compute order score of state sequences, not averaging, but with adj dupes removed

    A score of 0 means there's only one state.
    """

    scoresND = [] # scores with no adjacent duplicates

    for seqid in range(len(state_sequences)):
        logP = np.log(hmm.transmat_)
        pth = [x[0] for x in groupby(state_sequences[seqid])] # remove adjacent duplicates
        plen = len(pth)
        logPseq = 0
        for ii in range(plen-1):
            logPseq += logP[pth[ii],pth[ii+1]]
        score = logPseq
        scoresND.append(score)

    return np.array(scoresND)

def scoreOrderNA(hmm, state_sequences):
    """Compute order score of state sequences, not averaging

    A score of 0 means there's only one state.
    """

    scoresND = [] # scores with no adjacent duplicates

    for seqid in range(len(state_sequences)):
        logP = np.log(hmm.transmat_)
        pth = state_sequences[seqid]
        plen = len(pth)
        logPseq = 0
        for ii in range(plen-1):
            logPseq += logP[pth[ii],pth[ii+1]]
        score = logPseq
        scoresND.append(score)

    return np.array(scoresND)

def score_plen(hmm, state_sequences):
    """returns path length
    """
    plens = [] # scores with no adjacent duplicates

    for seqid in range(len(state_sequences)):
        pth = state_sequences[seqid]
        plen = len(pth)
        plens.append(plen)

    return np.array(plens)

def score_plenND(hmm, state_sequences):
    """returns path length, no adjacent duplicates
    """
    plens = [] # scores with no adjacent duplicates

    for seqid in range(len(state_sequences)):
        pth = [x[0] for x in groupby(state_sequences[seqid])] # remove adjacent duplicates
        plen = len(pth)
        plens.append(plen)

    return np.array(plens)

def score_SD(hmm, state_sequences):
    """returns State Diversity --- number of unique decoded states
    """
    plens = [] # scores with no adjacent duplicates

    for seqid in range(len(state_sequences)):
        pth = state_sequences[seqid]
        sd = len(set(pth))
        plens.append(sd)

    return np.array(plens)

def bigscore(hmm, state_sequences):
    scorefuncs = [score_SD,
                  score_plenND,
                  scoreOrderND,
                  scoreOrderNAND,
                  scoreOrderD,
                  scoreOrderNA,
                  score_plen]
    scores = []
    for scorefunc in scorefuncs:
        scores.append(scorefunc(hmm, state_sequences))

    # comboscore = ((-scoresND+scoresNAND-scoresD+scoresNA)/scoresplenND)+scoresSD
    comboscore = ((-scores[2]+scores[3]-scores[4]+scores[5])/scores[1])+scores[0]

    return comboscore, scores