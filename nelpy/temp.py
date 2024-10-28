# See https://github.com/NeuralEnsemble/elephant/blob/master/elephant/statistics.py
import nelpy as nel

def drop_last_bin(bst, inplace=False):
    """Drop the last bin from a BinnedSpikeTrainArray with a single epoch."""
    if bst.isempty:
        return bst
    assert bst.n_epochs == 1, "Not supported on BSTs with multiple epochs!"

    if inplace:
        out = bst
    else:
        from copy import deepcopy

        out = deepcopy(bst)

    out._support = bst._support.shrink(
        bst.ds, direction="stop"
    )  # shrink support(s) by one bin size, on right
    out._bin_centers = bst._bin_centers[
        :-1
    ]  # remove last bin center NB! this operates on last epoch only!
    out._binned_support[:, 1] = (
        bst._binned_support[:, 1] - 1
    )  # remove last bin from each epoch
    out._bins = bst._bins[:-1]
    out._data = bst._data[:, :-1]

    return out


def detect_ripples(eeg):
    """Detect sharp wave ripples (SWRs) from single channel eeg (AnalogSignalArray)."""

    # Maggie defines ripples by doing:
    #  (1) filter 150-250
    #  (2) hilbert envelope
    #  (3) smooth with Gaussian (4 ms SD)
    #  (4) 3.5 SD above the mean for 15 ms
    #  (5) full ripple defined as window back to mean

    assert (
        eeg.n_signals == 1
    ), "only single channel ripple detection currently supported!"

    # (1)
    ripple_eeg = nel.filtering.sosfiltfilt(eeg, fl=150, fh=250)
    # (2, 3)
    ripple_envelope = nel.utils.signal_envelope1D(ripple_eeg, sigma=0.004)
    # (4, 5)
    bounds, maxes, events = nel.utils.get_events_boundaries(
        x=ripple_envelope.data,
        PrimaryThreshold=ripple_envelope.mean() + 3.5 * ripple_envelope.std(),  # cm/s
        SecondaryThreshold=ripple_envelope.mean(),  # cm/s
        minThresholdLength=0.015,  # threshold crossing must be at least 15 ms long
        minLength=0.0,  # total ripple duration must be at least XXX ms long
        ds=1 / ripple_envelope.fs,
    )

    # convert bounds to time in seconds
    timebounds = ripple_envelope.time[bounds]

    # add 1/fs to stops for open interval
    timebounds[:, 1] += 1 / eeg.fs

    # create EpochArray with bounds
    ripple_epochs = nel.EpochArray(timebounds)

    # Adjust ripple centers to align to a peak
    # ripple_centers = np.floor((ripple_epochs.centers - eeg.time[0]) * eeg.fs).astype(
    #     int
    # )
    # ch = 7  # this was on some of Sibo's data, for CA1
    # adjusted_centers = [
    #     (p - 10) + np.argmax(eeg.data[ch, p - 10 : p + 10])
    #     for p in ripple_centers[1:-1].tolist()
    # ]

    return ripple_epochs


# Apply a z-score operation to one or several AnalogSignal objects.
#     The z-score operation subtracts the mean :math:`\\mu` of the signal, and
#     divides by its standard deviation :math:`\\sigma`:
#     .. math::
#          Z(x(t))= \\frac{x(t)-\\mu}{\\sigma}
#     If an AnalogSignal containing multiple signals is provided, the
#     z-transform is always calculated for each signal individually.
#     If a list of AnalogSignal objects is supplied, the mean and standard
#     deviation are calculated across all objects of the list. Thus, all list
#     elements are z-transformed by the same values of :math:`\\mu` and
#     :math:`\\sigma`. For AnalogSignals, each signal of the array is
#     treated separately across list elements. Therefore, the number of signals
#     must be identical for each AnalogSignal of the list.
#     Parameters
#     ----------
#     signal : neo.AnalogSignal or list of neo.AnalogSignal
#         Signals for which to calculate the z-score.
#     inplace : bool
#         If True, the contents of the input signal(s) is replaced by the
#         z-transformed signal. Otherwise, a copy of the original
#         AnalogSignal(s) is returned. Default: True
#     Returns
#     -------
#     neo.AnalogSignal or list of neo.AnalogSignal
#         The output format matches the input format: for each supplied
#         AnalogSignal object a corresponding object is returned containing
#         the z-transformed signal with the unit dimensionless.
#     Use Case
#     --------
#     You may supply a list of AnalogSignal objects, where each object in
#     the list contains the data of one trial of the experiment, and each signal
#     of the AnalogSignal corresponds to the recordings from one specific
#     electrode in a particular trial. In this scenario, you will z-transform the
#     signal of each electrode separately, but transform all trials of a given
#     electrode in the same way.
#     Examples
#     --------
#     >>> a = neo.AnalogSignal(
#     ...       np.array([1, 2, 3, 4, 5, 6]).reshape(-1,1)*mV,
#     ...       t_start=0*s, sampling_rate=1000*Hz)
#     >>> b = neo.AnalogSignal(
#     ...       np.transpose([[1, 2, 3, 4, 5, 6], [11, 12, 13, 14, 15, 16]])*mV,
#     ...       t_start=0*s, sampling_rate=1000*Hz)
#     >>> c = neo.AnalogSignal(
#     ...       np.transpose([[21, 22, 23, 24, 25, 26], [31, 32, 33, 34, 35, 36]])*mV,
#     ...       t_start=0*s, sampling_rate=1000*Hz)
#     >>> print zscore(a)
#     [[-1.46385011]
#      [-0.87831007]
#      [-0.29277002]
#      [ 0.29277002]
#      [ 0.87831007]
#      [ 1.46385011]] dimensionless
#     >>> print zscore(b)
#     [[-1.46385011 -1.46385011]
#      [-0.87831007 -0.87831007]
#      [-0.29277002 -0.29277002]
#      [ 0.29277002  0.29277002]
#      [ 0.87831007  0.87831007]
#      [ 1.46385011  1.46385011]] dimensionless
#     >>> print zscore([b,c])
#     [<AnalogSignal(array([[-1.11669108, -1.08361877],
#        [-1.0672076 , -1.04878252],
#        [-1.01772411, -1.01394628],
#        [-0.96824063, -0.97911003],
#        [-0.91875714, -0.94427378],
#        [-0.86927366, -0.90943753]]) * dimensionless, [0.0 s, 0.006 s],
#        sampling rate: 1000.0 Hz)>,
#        <AnalogSignal(array([[ 0.78170952,  0.84779261],
#        [ 0.86621866,  0.90728682],
#        [ 0.9507278 ,  0.96678104],
#        [ 1.03523694,  1.02627526],
#        [ 1.11974608,  1.08576948],
#        [ 1.20425521,  1.1452637 ]]) * dimensionless, [0.0 s, 0.006 s],
#        sampling rate: 1000.0 Hz)>]
