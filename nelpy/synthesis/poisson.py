import numpy as np


def GenerateSpikes(IntensityFunc, MaxRate, PositionFunc, TotalTime):
    # Start by generating spikes for a homogeneous Poisson process
    nHomogeneousSpikes = np.random.poisson(MaxRate * TotalTime)
    tHomogeneousSpikeTimes = np.random.uniform(0, TotalTime, nHomogeneousSpikes)
    tHomogeneousSpikeTimes.sort()

    if tHomogeneousSpikeTimes.size > 0:
        # Next, we need to evaluate intensity function at the locations/times of
        #  our generated spikes. inhomogeneousRate is an ndarray vector of the
        #  same length as the spike times.
        inhomogeneousRate = IntensityFunc(
            PositionFunc(tHomogeneousSpikeTimes), tHomogeneousSpikeTimes
        )

        # Then we'll compare the ratio of the inhomogeneousRates and the MaximumRate
        #  to a random number generator to decide when/when-not to delete.
        rnd = np.random.uniform(0, 1, inhomogeneousRate.size)
        tSpikeTimes = tHomogeneousSpikeTimes[inhomogeneousRate / MaxRate > rnd]
        return tSpikeTimes
    else:
        return tHomogeneousSpikeTimes


def generate_spikes_from_traj(
    binned_runidx,
    truepos,
    pfs,
    pfbincenters,
    pos_fs,
    const_firing_rate=False,
    verbose=False,
):

    # extract running trajectories from real data:
    run_ends = np.where(np.diff(binned_runidx) - 1)[0] + 1
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

    for ii in np.arange(len(runbdries) - 1):
        for nn in np.arange(numCells):
            traj_start_bin = binned_runidx[runbdries[ii]]
            traj_end_bin = binned_runidx[runbdries[ii + 1] - 1]  # inclusive
            # print("Trajectory {0} length is {1} bins, or {2:2.2f} seconds.".format(ii,traj_end_bin-traj_start_bin + 1, (traj_end_bin-traj_start_bin + 1)/pos_fs))
            t_offset = traj_start_bin / pos_fs  # in seconds
            posvect = truepos[runbdries[ii] : runbdries[ii + 1]]
            TrajDuration = len(posvect) / pos_fs
            # print("Traj duration {0:2.2f}".format(TrajDuration))
            tvect = np.linspace(
                0, TrajDuration, len(posvect)
            )  # WARNING! This does not work when traj_len == 1, but this should never happen anyway

            posFun = lambda t: np.interp(t, tvect, posvect)
            if const_firing_rate:
                PlaceFieldRate = (
                    lambda x: 0 * x + 10
                )  # np.interp(x,pfbincenters,pfs[nn,:])
            else:
                PlaceFieldRate = lambda x: np.interp(x, pfbincenters, pfs[nn, :])
            maxrate = pfs[nn, :].max() + 25

            SpikeRasters[ii][nn] = np.round(
                (
                    t_offset
                    + GenerateSpikes(
                        lambda x, t: PlaceFieldRate(x), maxrate, posFun, TrajDuration
                    )
                )
                * fs
            )

            # DEBUG:
            tempsum += len(SpikeRasters[ii][nn])
            tempdur += TrajDuration
    if verbose:
        print(
            "true average spike rate synthesized (across all cells), before place field estimation: {0:1.2f} Hz".format(
                tempsum / tempdur
            )
        )

    # consolidate spike rasters (not stratified by trajectory)
    cons_st_array = []
    for nn in np.arange(numCells):
        st = np.zeros(0)
        for ii in np.arange(NumTrajectories):
            st = np.hstack((st, SpikeRasters[ii][nn]))
        st = np.sort(st)
        cons_st_array.append(st)

    return cons_st_array
