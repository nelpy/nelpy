import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
import warnings

from scipy import signal

from .helpers import RasterLabelData
from . import utils  # import plotting/utils
from .. import auxiliary
from .. import core

__all__ = [
    "plot",
    "plot2d",
    "colorline",
    "plot_tuning_curves1D",
    "psdplot",
    "overviewstrip",
    "imagesc",
    "matshow",
    "epochplot",
    "rasterplot",
    "rastercountplot",
    "_plot_ratemap",
]


def colorline(x, y, cmap=None, cm_range=(0, 0.7), **kwargs):
    """Colorline plots a trajectory of (x,y) points with a colormap"""

    # plt.plot(x, y, '-k', zorder=1)
    # plt.scatter(x, y, s=40, c=plt.cm.RdBu(np.linspace(0,1,40)), zorder=2, edgecolor='k')

    assert len(cm_range) == 2, "cm_range must have (min, max)"
    assert len(x) == len(y), "x and y must have the same number of elements!"

    ax = kwargs.get("ax", plt.gca())
    lw = kwargs.get("lw", 2)
    if cmap is None:
        cmap = plt.cm.Blues_r

    t = np.linspace(cm_range[0], cm_range[1], len(x))

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc = LineCollection(segments, cmap=cmap, norm=plt.Normalize(0, 1), zorder=50)
    lc.set_array(t)
    lc.set_linewidth(lw)

    ax.add_collection(lc)

    return lc


def _plot_ratemap(
    ratemap, ax=None, normalize=False, pad=None, unit_labels=None, fill=True, color=None
):
    """
    WARNING! This function is not complete, and hence 'private',
    and may be moved somewhere else later on.

    If pad=0 then the y-axis is assumed to be firing rate
    """
    if ax is None:
        ax = plt.gca()

    # xmin = ratemap.bins[0]
    # xmax = ratemap.bins[-1]
    # xvals = ratemap.bin_centers
    if unit_labels is None:
        unit_labels = ratemap.unit_ids
    ratemap = ratemap.ratemap_

    if pad is None:
        pad = ratemap.mean() / 2

    n_units, n_ext = ratemap.shape

    if normalize:
        peak_firing_rates = ratemap.max(axis=1)
        ratemap = (ratemap.T / peak_firing_rates).T

    # determine max firing rate
    # max_firing_rate = ratemap.max()
    xvals = np.arange(n_ext)

    for unit, curve in enumerate(ratemap):
        if color is None:
            line = ax.plot(
                xvals, unit * pad + curve, zorder=int(10 + 2 * n_units - 2 * unit)
            )
        else:
            line = ax.plot(
                xvals,
                unit * pad + curve,
                zorder=int(10 + 2 * n_units - 2 * unit),
                color=color,
            )
        if fill:
            # Get the color from the current curve
            fillcolor = line[0].get_color()
            ax.fill_between(
                xvals,
                unit * pad,
                unit * pad + curve,
                alpha=0.3,
                color=fillcolor,
                zorder=int(10 + 2 * n_units - 2 * unit - 1),
            )

    # ax.set_xlim(xmin, xmax)
    if pad != 0:
        yticks = np.arange(n_units) * pad + 0.5 * pad
        ax.set_yticks(yticks)
        ax.set_yticklabels(unit_labels)
        ax.set_xlabel("external variable")
        ax.set_ylabel("unit")
        utils.no_yticks(ax)
        utils.clear_left(ax)
    else:
        if normalize:
            ax.set_ylabel("normalized firing rate")
        else:
            ax.set_ylabel("firing rate [Hz]")
        ax.set_ylim(0)

    utils.clear_top(ax)
    utils.clear_right(ax)

    return ax


def plot_tuning_curves1D(
    ratemap,
    ax=None,
    normalize=False,
    pad=None,
    unit_labels=None,
    fill=True,
    color=None,
    alpha=0.3,
):
    """
    WARNING! This function is not complete, and hence 'private',
    and may be moved somewhere else later on.

    If pad=0 then the y-axis is assumed to be firing rate
    """
    if ax is None:
        ax = plt.gca()

    if isinstance(ratemap, auxiliary.TuningCurve1D) | isinstance(
        ratemap, auxiliary._tuningcurve.TuningCurve1D
    ):
        xmin = ratemap.bins[0]
        xmax = ratemap.bins[-1]
        xvals = ratemap.bin_centers
        if unit_labels is None:
            unit_labels = ratemap.unit_labels
        ratemap = ratemap.ratemap
    else:
        raise NotImplementedError

    if pad is None:
        pad = ratemap.mean() / 2

    n_units, n_ext = ratemap.shape

    if normalize:
        peak_firing_rates = ratemap.max(axis=1)
        ratemap = (ratemap.T / peak_firing_rates).T

    # determine max firing rate
    # max_firing_rate = ratemap.max()

    if xvals is None:
        xvals = np.arange(n_ext)
    if xmin is None:
        xmin = xvals[0]
    if xmax is None:
        xmax = xvals[-1]

    for unit, curve in enumerate(ratemap):
        if color is None:
            line = ax.plot(
                xvals, unit * pad + curve, zorder=int(10 + 2 * n_units - 2 * unit)
            )
        else:
            line = ax.plot(
                xvals,
                unit * pad + curve,
                zorder=int(10 + 2 * n_units - 2 * unit),
                color=color,
            )
        if fill:
            # Get the color from the current curve
            fillcolor = line[0].get_color()
            ax.fill_between(
                xvals,
                unit * pad,
                unit * pad + curve,
                alpha=alpha,
                color=fillcolor,
                zorder=int(10 + 2 * n_units - 2 * unit - 1),
            )

    ax.set_xlim(xmin, xmax)
    if pad != 0:
        yticks = np.arange(n_units) * pad + 0.5 * pad
        ax.set_yticks(yticks)
        ax.set_yticklabels(unit_labels)
        ax.set_xlabel("external variable")
        ax.set_ylabel("unit")
        utils.no_yticks(ax)
        utils.clear_left(ax)
    else:
        if normalize:
            ax.set_ylabel("normalized firing rate")
        else:
            ax.set_ylabel("firing rate [Hz]")
        ax.set_ylim(0)

    utils.clear_top(ax)
    utils.clear_right(ax)

    return ax


def spectrogram(data, *, h):
    """
    Compute a spectrogram with consecutive Fourier transforms.
    Spectrograms can be used as a way of visualizing the change of a
    nonstationary signal's frequency content over time.
    Parameters
    ----------
    x : array_like
        Time series of measurement values
    fs : float, optional
        Sampling frequency of the `x` time series. Defaults to 1.0.
    window : str or tuple or array_like, optional
        Desired window to use. See `get_window` for a list of windows
        and required parameters. If `window` is array_like it will be
        used directly as the window and its length must be nperseg.
        Defaults to a Tukey window with shape parameter of 0.25.
    nperseg : int, optional
        Length of each segment. Defaults to None, but if window is str or
        tuple, is set to 256, and if window is array_like, is set to the
        length of the window.
    noverlap : int, optional
        Number of points to overlap between segments. If `None`,
        ``noverlap = nperseg // 8``. Defaults to `None`.
    nfft : int, optional
        Length of the FFT used, if a zero padded FFT is desired. If
        `None`, the FFT length is `nperseg`. Defaults to `None`.
    detrend : str or function or `False`, optional
        Specifies how to detrend each segment. If `detrend` is a
        string, it is passed as the `type` argument to the `detrend`
        function. If it is a function, it takes a segment and returns a
        detrended segment. If `detrend` is `False`, no detrending is
        done. Defaults to 'constant'.
    return_onesided : bool, optional
        If `True`, return a one-sided spectrum for real data. If
        `False` return a two-sided spectrum. Note that for complex
        data, a two-sided spectrum is always returned.
    scaling : { 'density', 'spectrum' }, optional
        Selects between computing the power spectral density ('density')
        where `Sxx` has units of V**2/Hz and computing the power
        spectrum ('spectrum') where `Sxx` has units of V**2, if `x`
        is measured in V and `fs` is measured in Hz. Defaults to
        'density'.
    axis : int, optional
        Axis along which the spectrogram is computed; the default is over
        the last axis (i.e. ``axis=-1``).
    mode : str, optional
        Defines what kind of return values are expected. Options are
        ['psd', 'complex', 'magnitude', 'angle', 'phase']. 'complex' is
        equivalent to the output of `stft` with no padding or boundary
        extension. 'magnitude' returns the absolute magnitude of the
        STFT. 'angle' and 'phase' return the complex angle of the STFT,
        with and without unwrapping, respectively.
    Returns
    -------
    f : ndarray
        Array of sample frequencies.
    t : ndarray
        Array of segment times.
    Sxx : ndarray
        Spectrogram of x. By default, the last axis of Sxx corresponds
        to the segment times.

    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.spectrogram.html
    """
    raise NotImplementedError("plotting.spectrogram() does not exist yet!")


def psdplot(
    data,
    *,
    fs=None,
    window=None,
    nfft=None,
    detrend="constant",
    return_onesided=True,
    scaling="density",
    ax=None,
):
    """
    Plot the power spectrum of a regularly-sampled time-domain signal.

    Parameters
    ----------
    data : RegularlySampledAnalogSignalArray
        The input signal to analyze. Must be a 1D regularly sampled signal.
    fs : float, optional
        Sampling frequency of the time series in Hz. Defaults to data.fs if available.
    window : str or tuple or array_like, optional
        Desired window to use. See scipy.signal.get_window for options. If an array, used directly as the window. Defaults to None ('boxcar').
    nfft : int, optional
        Length of the FFT used. If None, the length of data will be used.
    detrend : str or function, optional
        Specifies how to detrend data prior to computing the spectrum. If a string, passed as the type argument to detrend. If a function, should return a detrended array. Defaults to 'constant'.
    return_onesided : bool, optional
        If True, return a one-sided spectrum for real data. If False, return a two-sided spectrum. For complex data, always returns two-sided spectrum.
    scaling : {'density', 'spectrum'}, optional
        Selects between computing the power spectral density ('density', units V**2/Hz) and the power spectrum ('spectrum', units V**2). Defaults to 'density'.
    ax : matplotlib.axes.Axes, optional
        Axis to plot on. If None, creates a new figure and axis.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axis with the plotted power spectrum.

    Examples
    --------
    >>> from nelpy.plotting.core import psdplot
    >>> ax = psdplot(my_signal)
    """

    if ax is None:
        ax = plt.gca()

    if isinstance(data, core.RegularlySampledAnalogSignalArray):
        if fs is None:
            fs = data.fs
        if fs is None:
            raise ValueError(
                "The sampling rate fs cannot be inferred, and must be specified manually!"
            )
        if data.n_signals > 1:
            raise NotImplementedError(
                "more than one signal is not yet supported for psdplot!"
            )
        else:
            data = data.data.squeeze()
    else:
        raise NotImplementedError(
            "datatype {} not yet supported by psdplot!".format(str(type(data)))
        )

    kwargs = {
        "x": data,
        "fs": fs,
        "window": window,
        "nfft": nfft,
        "detrend": detrend,
        "return_onesided": return_onesided,
        "scaling": scaling,
    }

    f, Pxx_den = signal.periodogram(**kwargs)

    if scaling == "density":
        ax.semilogy(f, np.sqrt(Pxx_den))
        ax.set_ylabel("PSD [V**2/Hz]")
    elif scaling == "spectrum":
        ax.semilogy(f, np.sqrt(Pxx_den))
        ax.set_ylabel("Linear spectrum [V RMS]")
    ax.set_xlabel("frequency [Hz]")

    return ax


def imagesc(x=None, y=None, data=None, *, ax=None, large=False, **kwargs):
    """Plots a 2D matrix / image similar to Matlab's imagesc.

    Parameters
    ----------
    x : array-like, optional
        x values (cols)
    y : array-like, optional
        y-values (rows)
    data : ndarray of shape (Nrows, Ncols)
        matrix to visualize
    ax : matplotlib axis, optional
        Plot in given axis; if None creates a new figure
    large : bool
        If True, use ModestImage instead of mpl.AxesImage that supports
        much larger images, but the 'extent' does not work properly yet.
    kwargs :
        Other keyword arguments are passed to main imagesc() call

    Returns
    -------
    ax : matplotlib axis
        Axis object with plot data.
    image : matplotlib image

    Examples
    -------
    Plot a simple matrix using imagesc

        >>> x = np.linspace(-100, -10, 10)
        >>> y = np.array([-8, -3.0])
        >>> data = np.random.randn(y.size, x.size)
        >>> imagesc(x, y, data)
    or
        >>> imagesc(data)

    Adding a colorbar

        >>> ax, img = imagesc(data)
        >>> from mpl_toolkits.axes_grid1 import make_axes_locatable
        >>> divider = make_axes_locatable(ax)
        >>> cax = divider.append_axes("right", size="3.5%", pad=0.1)
        >>> cb = plt.colorbar(img, cax=cax)
        >>> npl.utils.no_yticks(cax)
    """

    def extents(f):
        if len(f) > 1:
            delta = f[1] - f[0]
        else:
            delta = 1
        return [f[0] - delta / 2, f[-1] + delta / 2]

    if ax is None:
        ax = plt.gca()
    if data is None:
        if x is None:  # no args
            raise ValueError(
                "Unknown input. Usage imagesc(x, y, data) or imagesc(data)."
            )
        elif y is None:  # only one arg, so assume it to be data
            data = x
            x = np.arange(data.shape[1])
            y = np.arange(data.shape[0])
        else:  # x and y, but no data
            raise ValueError(
                "Unknown input. Usage imagesc(x, y, data) or imagesc(data)."
            )

    if data.ndim != 2:
        raise TypeError("data must be 2 dimensional")

    if not large:
        # Matplotlib imshow
        image = ax.imshow(
            data,
            aspect="auto",
            interpolation="none",
            extent=extents(x) + extents(y),
            origin="lower",
            **kwargs,
        )
    else:
        # ModestImage imshow for large images, but 'extent' is still not working well
        image = utils.imshow(
            axes=ax,
            X=data,
            aspect="auto",
            interpolation="none",
            extent=extents(x) + extents(y),
            origin="lower",
            **kwargs,
        )

    return ax, image


def plot(obj, *args, **kwargs):
    """Docstring goes here."""

    ax = kwargs.pop("ax", None)
    autoscale = kwargs.pop("autoscale", True)
    xlabel = kwargs.pop("xlabel", None)
    ylabel = kwargs.pop("ylabel", None)

    if ax is None:
        ax = plt.gca()

    if isinstance(obj, core.RegularlySampledAnalogSignalArray):
        if obj.n_signals == 1:
            label = kwargs.pop("label", None)
            for ii, (abscissa_vals, data) in enumerate(
                zip(
                    obj._intervaltime.plot_generator(),
                    obj._intervaldata.plot_generator(),
                )
            ):
                ax.plot(
                    abscissa_vals,
                    data.T,
                    label=label if ii == 0 else "_nolegend_",
                    *args,
                    **kwargs,
                )
        elif obj.n_signals > 1:
            # TODO: intercept when any color is requested. This could happen
            # multiple ways, such as plt.plot(x, '-r') or plt.plot(x, c='0.7')
            # or plt.plot(x, color='red'), and maybe some others? Probably have
            # dig into the matplotlib code to see how they parse this and do
            # conflict resolution... Update: they use the last specified color.
            # but I still need to know how to detect a color that was passed in
            # the *args part, e.g., '-r'

            color = kwargs.pop("color", None)
            carg = kwargs.pop("c", None)

            if color is not None and carg is not None:
                # TODO: fix this so that a warning is issued, not raised
                raise ValueError("saw kwargs ['c', 'color']")
                # raise UserWarning("saw kwargs ['c', 'color'] which are all aliases for 'color'.  Kept value from 'color'")
            if carg:
                color = carg

            if not color:
                colors = []
                for ii in range(obj.n_signals):
                    (line,) = ax.plot(0, 0.5)
                    colors.append(line.get_color())
                    line.remove()

                for ee, (abscissa_vals, data) in enumerate(
                    zip(
                        obj._intervaltime.plot_generator(),
                        obj._intervaldata.plot_generator(),
                    )
                ):
                    if ee > 0:
                        kwargs["label"] = "_nolegend_"
                    for ii, snippet in enumerate(data):
                        ax.plot(
                            abscissa_vals, snippet, *args, color=colors[ii], **kwargs
                        )
            else:
                kwargs["color"] = color
                for ee, (abscissa_vals, data) in enumerate(
                    zip(
                        obj._intervaltime.plot_generator(),
                        obj._intervaldata.plot_generator(),
                    )
                ):
                    if ee > 0:
                        kwargs["label"] = "_nolegend_"
                    for ii, snippet in enumerate(data):
                        ax.plot(abscissa_vals, snippet, *args, **kwargs)

        if xlabel is None:
            xlabel = obj._abscissa.label
        if ylabel is None:
            ylabel = obj._ordinate.label
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    else:  # if we didn't handle it yet, just pass it through to matplotlib...
        ax.plot(obj, *args, **kwargs)

    if autoscale:
        xmin = np.inf
        xmax = -np.inf
        ymin = np.inf
        ymax = -np.inf
        for child in ax.get_children():
            try:
                cxmin, cymin = np.min(child.get_xydata(), axis=0)
                cxmax, cymax = np.max(child.get_xydata(), axis=0)
                if cxmin < xmin:
                    xmin = cxmin
                if cymin < ymin:
                    ymin = cymin
                if cxmax > xmax:
                    xmax = cxmax
                if cymax > ymax:
                    ymax = cymax
            except Exception:
                pass
        ax.set_xlim(xmin, xmax)


def plot_old(
    npl_obj,
    data=None,
    *,
    ax=None,
    mew=None,
    color=None,
    mec=None,
    markerfacecolor=None,
    **kwargs,
):
    """Plot an array-like object on an EpochArray.

    Parameters
    ----------
    npl_obj : nelpy.EpochArray or nelpy.AnalogSignal
        EpochArray on which the data is defined or AnalogSignal with data
    data : array-like
        Data to plot on y axis; must be of size (epocharray.n_epochs,).
    ax : axis object, optional
        Plot in given axis; if None creates a new figure
    mew : float, optional
        Marker edge width, default is equal to lw.
    color : matplotlib color, optional
        Trace color.
    mec : matplotlib color, optional
        Marker edge color, default is equal to color.
    kwargs :
        Other keyword arguments are passed to main plot() call

    Returns
    -------
    ax : matplotlib axis
        Axis object with plot data.

    Examples
    --------
    Plot a simple 5-element list on an EpochArray:

        >>> ep = EpochArray([[3, 4], [5, 8], [10, 12], [16, 20], [22, 23]])
        >>> data = [3, 4, 2, 5, 2]
        >>> npl.plot(ep, data)

    Hide the markers and change the linewidth:

        >>> ep = EpochArray([[3, 4], [5, 8], [10, 12], [16, 20], [22, 23]])
        >>> data = [3, 4, 2, 5, 2]
        >>> npl.plot(ep, data, ms=0, lw=3)
    """

    if ax is None:
        ax = plt.gca()
    if mec is None:
        mec = color
    if markerfacecolor is None:
        markerfacecolor = "w"

    if isinstance(npl_obj, np.ndarray):
        ax.plot(npl_obj, mec=mec, markerfacecolor=markerfacecolor, **kwargs)

    # TODO: better solution for this? we could just iterate over the epochs and
    # plot them but that might take up too much time since a copy is being made
    # each iteration?
    if isinstance(npl_obj, core.RegularlySampledAnalogSignalArray):
        # Get the colors from the current color cycle
        if npl_obj.n_signals > 1:
            colors = []
            lines = []
            for ii in range(npl_obj.n_signals):
                (line,) = ax.plot(0, 0.5)
                lines.append(line)
                colors.append(line.get_color())
                # line.remove()
            for line in lines:
                line.remove()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if not npl_obj.labels:
                for segment in npl_obj:
                    if color is not None:
                        ax.plot(
                            segment._time,
                            segment._data_colsig,
                            color=color,
                            mec=mec,
                            markerfacecolor=markerfacecolor,
                            **kwargs,
                        )
                    else:
                        ax.plot(
                            segment._time,
                            segment._data_colsig,
                            # color=colors[ii],
                            mec=mec,
                            markerfacecolor=markerfacecolor,
                            **kwargs,
                        )
            else:  # there are labels
                if npl_obj.n_signals > 1:
                    for ii, segment in enumerate(npl_obj):
                        for signal, label in zip(segment._data_rowsig, npl_obj.labels):
                            if color is not None:
                                ax.plot(
                                    segment._time,
                                    signal,
                                    color=color,
                                    mec=mec,
                                    markerfacecolor=markerfacecolor,
                                    label=label if ii == 0 else "_nolegend_",
                                    **kwargs,
                                )
                            else:  # color(s) have not been specified, use color cycler
                                ax.plot(
                                    segment._time,
                                    signal,
                                    # color=colors[ii],
                                    mec=mec,
                                    markerfacecolor=markerfacecolor,
                                    label=label if ii == 0 else "_nolegend_",
                                    **kwargs,
                                )
                else:  # only one signal
                    for ii, segment in enumerate(npl_obj):
                        if not npl_obj.labels:
                            label = None
                        else:
                            label = npl_obj.labels
                        if color is not None:
                            ax.plot(
                                segment._time,
                                segment._data_colsig,
                                color=color,
                                mec=mec,
                                markerfacecolor=markerfacecolor,
                                label=label if ii == 0 else "_nolegend_",
                                **kwargs,
                            )
                        else:
                            ax.plot(
                                segment._time,
                                segment._data_colsig,
                                # color=color,
                                mec=mec,
                                markerfacecolor=markerfacecolor,
                                label=label if ii == 0 else "_nolegend_",
                                **kwargs,
                            )

    if isinstance(npl_obj, core.IntervalArray):
        epocharray = npl_obj
        if epocharray.n_intervals != len(data):
            raise ValueError("epocharray and data must have the same length")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for epoch, val in zip(epocharray, data):
                ax.plot(
                    [epoch.start, epoch.stop],
                    [val, val],
                    "-o",
                    color=color,
                    mec=mec,
                    markerfacecolor=markerfacecolor,
                    # lw=lw,
                    # mew=mew,
                    **kwargs,
                )

        # ax.set_ylim([np.array(data).min()-0.5, np.array(data).max()+0.5])

    return ax


def plot2d(
    npl_obj,
    data=None,
    *,
    ax=None,
    mew=None,
    color=None,
    mec=None,
    markerfacecolor=None,
    **kwargs,
):
    """
    THIS SHOULD BE UPDATED! VERY RUDIMENTARY AT THIS STAGE
    """

    if ax is None:
        ax = plt.gca()
    if mec is None:
        mec = color
    if markerfacecolor is None:
        markerfacecolor = "w"

    if isinstance(npl_obj, np.ndarray):
        ax.plot(npl_obj, mec=mec, markerfacecolor=markerfacecolor, **kwargs)

    # TODO: better solution for this? we could just iterate over the epochs and
    # plot them but that might take up too much time since a copy is being made
    # each iteration?
    if isinstance(npl_obj, core.RegularlySampledAnalogSignalArray):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for segment in npl_obj:
                if color is not None:
                    ax.plot(
                        segment[:, 0]._data_colsig,
                        segment[:, 1]._data_colsig,
                        color=color,
                        mec=mec,
                        markerfacecolor="w",
                        **kwargs,
                    )
                else:
                    ax.plot(
                        segment[:, 0]._data_colsig,
                        segment[:, 1]._data_colsig,
                        # color=color,
                        mec=mec,
                        markerfacecolor="w",
                        **kwargs,
                    )

    if isinstance(npl_obj, core.PositionArray):
        xlim, ylim = npl_obj.xlim, npl_obj.ylim
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
    return ax


def imshow(data, *, ax=None, interpolation=None, **kwargs):
    """Docstring goes here."""

    # set default interpolation mode to 'none'
    if interpolation is None:
        interpolation = "none"


def matshow(data, *, ax=None, **kwargs):
    """Docstring goes here."""

    # Sort out default values for the parameters
    if ax is None:
        ax = plt.gca()

    # Handle different types of input data
    if isinstance(data, core.BinnedSpikeTrainArray):
        # TODO: split by epoch, and plot matshows in same row, but with
        # a small gap to indicate discontinuities. How about slicing
        # then? Or slicing within an epoch?
        ax.matshow(data.data, **kwargs)
        ax.set_xlabel("time")
        ax.set_ylabel("unit")
        warnings.warn("Automatic x-axis formatting not yet implemented")
    else:
        raise NotImplementedError(
            "matshow({}) not yet supported".format(str(type(data)))
        )

    return ax


def comboplot(*, ax=None, raster=None, analog=None, events=None):
    """Combo plot (consider better name) showing spike / state raster
    with additional analog signals, such as LFP or velocity, and also
    possibly with events. Here, the benefit is to have the figure and
    axes created automatically, in addition to prettification, as well
    as axis-linking. I don't know if we will really call this plot often
    though, so may be more of a gimmick?
    """

    # Sort out default values for the parameters
    if ax is None:
        ax = plt.gca()

    raise NotImplementedError("comboplot() not implemented yet")

    return ax


def occupancy():
    """
    Compute and/or plot the occupancy (time spent) in each spatial bin.

    Returns
    -------
    occupancy : np.ndarray
        Array of occupancy values for each bin.

    Notes
    -----
    This function is not yet implemented.
    """
    raise NotImplementedError("occupancy() not implemented yet")


def overviewstrip(
    epochs, *, ax=None, lw=5, solid_capstyle="butt", label=None, **kwargs
):
    """Plot an epoch array similar to vscode scrollbar, to show gaps in e.g.
    matshow plots. TODO: complete me.

    This can also be nice, for example, to implement the Kloosterman 2012
    online vs offline strips above several of the plots.
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    if ax is None:
        ax = plt.gca()

    divider = make_axes_locatable(ax)
    ax_ = divider.append_axes("top", size=0.2, pad=0.05)

    for epoch in epochs:
        ax_.plot(
            [epoch.start, epoch.stop],
            [1, 1],
            lw=lw,
            solid_capstyle=solid_capstyle,
            **kwargs,
        )

    if label is not None:
        ax_.set_yticks([1])
        ax_.set_yticklabels([label])
    else:
        ax_.set_yticks([])

    utils.no_yticks(ax_)
    utils.clear_left(ax_)
    utils.clear_right(ax_)
    utils.clear_top_bottom(ax_)

    ax_.set_xlim(ax.get_xlim())


def rastercountplot(spiketrain, nbins=50, **kwargs):
    plt.figure(figsize=(14, 6))
    gs = gridspec.GridSpec(2, 1, hspace=0.01, height_ratios=[0.2, 0.8])
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])

    color = kwargs.get("color", None)
    if color is None:
        color = "0.4"

    ds = (spiketrain.support.stop - spiketrain.support.start) / nbins
    flattened = spiketrain.bin(ds=ds).flatten()
    steps = np.squeeze(flattened.data)
    stepsx = np.linspace(
        spiketrain.support.start, spiketrain.support.stop, num=flattened.n_bins
    )

    #     ax1.plot(stepsx, steps, drawstyle='steps-mid', color='none');
    ax1.set_ylim([-0.5, np.max(steps) + 1])
    rasterplot(spiketrain, ax=ax2, **kwargs)

    utils.clear_left_right(ax1)
    utils.clear_top_bottom(ax1)
    utils.clear_top(ax2)

    ax1.fill_between(stepsx, steps, step="mid", color=color)

    utils.sync_xlims(ax1, ax2)

    return ax1, ax2


def rasterplot(
    data,
    *,
    cmap=None,
    color=None,
    ax=None,
    lw=None,
    lh=None,
    vertstack=None,
    labels=None,
    cmap_lo=0.25,
    cmap_hi=0.75,
    **kwargs,
):
    """Make a raster plot from a SpikeTrainArray object.

    Parameters
    ----------
    data : nelpy.SpikeTrainArray object
    cmap : matplotlib colormap, optional
    color: matplotlib color, optional
        Plot color; default is '0.25'
    ax : axis object, optional
        Plot in given axis. If None, plots on current axes
    lw : float, optional
        Linewidth, default value of lw=1.5.
    lh : float, optional
        Line height, default value of 0.95
    vertstack : Stack units in vertically adjacent positions, optional
        Default is to plot units in absolute positions according
        to their unit_ids
    labels : Labels for input data units, optional
        If not specified, default is to use the unit_labels from the
        SpikeTrainArray input. See SpikeTrainArray docstring for
        default behavior of unit_labels
    kwargs :
        Other keyword arguments are passed to main vlines() call

    Returns
    -------
    ax : matplotlib axis
        Axis object with plot data.

    Examples
    --------
    Instantiate a SpikeTrainArray and create a raster plot
        >>> stdata1 = [1, 2, 4, 5, 6, 10, 20]
        >>> stdata2 = [3, 4, 4.5, 5, 5.5, 19]
        >>> stdata3 = [5, 12, 14, 15, 16, 18, 22, 23, 24]
        >>> stdata4 = [5, 12, 14, 15, 16, 18, 23, 25, 32]

        >>> sta1 = nelpy.SpikeTrainArray([stdata1, stdata2, stdata3,
                                          stdata4, stdata1+stdata4],
                                          fs=5, unit_ids=[1,2,3,4,6])
        >>> ax = rasterplot(sta1, color="cyan", lw=2, lh=2)

    Instantiate another SpikeTrain Array, stack units, and specify labels.
    Note that the user-specified labels in the call to raster() will be
    shown instead of the unit_labels associated with the input data
        >>> sta3 = nelpy.SpikeTrainArray([stdata1, stdata4, stdata2+stdata3],
                                         support=ep1, fs=5, unit_ids=[10,5,12],
                                         unit_labels=['some', 'more', 'cells'])
        >>> rasterplot(sta3, color=plt.cm.Blues, lw=2, lh=2, vertstack=True,
                   labels=['units', 'of', 'interest'])
    """

    # Sort out default values for the parameters
    if ax is None:
        ax = plt.gca()
    if cmap is None and color is None:
        color = "0.25"
    if lw is None:
        lw = 1.5
    if lh is None:
        lh = 0.95
    if vertstack is None:
        vertstack = False

    firstplot = False
    if not ax.findobj(match=RasterLabelData):
        firstplot = True
        ax.add_artist(RasterLabelData())

    # override labels
    if labels is not None:
        series_labels = labels
    else:
        series_labels = []

    hh = lh / 2.0  # half the line height

    # Handle different types of input data
    if isinstance(data, core.EventArray):
        label_data = ax.findobj(match=RasterLabelData)[0].label_data
        serieslist = [-np.inf for element in data.series_ids]
        # no override labels so use unit_labels from input
        if not series_labels:
            series_labels = data.series_labels

        if firstplot:
            if vertstack:
                minunit = 1
                maxunit = data.n_series
                serieslist = range(1, data.n_series + 1)
            else:
                minunit = np.array(data.series_ids).min()
                maxunit = np.array(data.series_ids).max()
                serieslist = data.series_ids
        # see if any of the series_ids has already been plotted. If so,
        # then merge
        else:
            for idx, series_id in enumerate(data.series_ids):
                if series_id in label_data.keys():
                    position, _ = label_data[series_id]
                    serieslist[idx] = position
                else:  # unit not yet plotted
                    if vertstack:
                        serieslist[idx] = 1 + max(
                            int(ax.get_yticks()[-1]), max(serieslist)
                        )
                    else:
                        warnings.warn(
                            "Spike trains may be plotted in "
                            "the same vertical position as "
                            "another unit"
                        )
                        serieslist[idx] = data.series_ids[idx]

        if firstplot:
            minunit = int(minunit)
            maxunit = int(maxunit)
        else:
            (prev_ymin, prev_ymax) = ax.findobj(match=RasterLabelData)[0].yrange
            minunit = int(np.min([np.ceil(prev_ymin), np.min(serieslist)]))
            maxunit = int(np.max([np.floor(prev_ymax), np.max(serieslist)]))

        yrange = (minunit - 0.5, maxunit + 0.5)

        if cmap is not None:
            color_range = range(data.n_series)
            # TODO: if we go from 0 then most colormaps are invisible at one end of the spectrum
            colors = cmap(np.linspace(cmap_lo, cmap_hi, data.n_series))
            for series_ii, series, color_idx in zip(serieslist, data.data, color_range):
                ax.vlines(
                    series,
                    series_ii - hh,
                    series_ii + hh,
                    colors=colors[color_idx],
                    lw=lw,
                    **kwargs,
                )
        else:  # use a constant color:
            for series_ii, series in zip(serieslist, data.data):
                ax.vlines(
                    series,
                    series_ii - hh,
                    series_ii + hh,
                    colors=color,
                    lw=lw,
                    **kwargs,
                )

        # get existing label data so we can set some attributes
        rld = ax.findobj(match=RasterLabelData)[0]

        ax.set_ylim(yrange)
        rld.yrange = yrange

        for series_id, loc, label in zip(data.series_ids, serieslist, series_labels):
            rld.label_data[series_id] = (loc, label)
        serieslocs = []
        serieslabels = []
        for loc, label in label_data.values():
            serieslocs.append(loc)
            serieslabels.append(label)
        ax.set_yticks(serieslocs)
        ax.set_yticklabels(serieslabels)

    else:
        raise NotImplementedError(
            "plotting {} not yet supported".format(str(type(data)))
        )
    return ax


def epochplot(
    epochs,
    data=None,
    *,
    ax=None,
    height=None,
    fc="0.5",
    ec="0.5",
    alpha=0.5,
    hatch="",
    label=None,
    hc=None,
    **kwargs,
):
    """
    Plot epochs as vertical bars on a plot.

    Parameters
    ----------
    epochs : nelpy.EpochArray
        EpochArray object to plot
    data : array-like, optional
        Data to plot on y axis; must be of size (epocharray.n_epochs,).
    ax : axis object, optional
        Plot in given axis; if None creates a new figure
    height : float, optional
        Height of the bars; default is 1.0
    fc : matplotlib color, optional
        Face color; default is '0.5'
    ec : matplotlib color, optional
        Edge color; default is '0.5'
    alpha : float, optional
        Transparency; default is 0.5
    hatch : str, optional
        Hatch pattern; default is ''
    label : str, optional
        Label for the plot
    hc : matplotlib color, optional
        Hatch color; default is None
    kwargs :
        Other keyword arguments are passed to main axvspan() call

    Returns
    -------
    ax : matplotlib axis
        Axis object with plot data.

    """
    if ax is None:
        ax = plt.gca()

    # do fixed-value-on-epoch plot if data is not None
    if data is not None:
        if epochs.n_intervals != len(data):
            raise ValueError("epocharray and data must have the same length")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for epoch, val in zip(epochs, data):
                ax.plot([epoch.start, epoch.stop], [val, val], "-o", **kwargs)
        return ax

    ymin, ymax = ax.get_ylim()
    xmin, xmax = ax.get_xlim()
    if height is None:
        height = ymax - ymin

    if hc is not None:
        try:
            hc_before = mpl.rcParams["hatch.color"]
            mpl.rcParams["hatch.color"] = hc
        except KeyError:
            warnings.warn("Hatch color not supported for matplotlib <2.0")

    for ii, (start, stop) in enumerate(zip(epochs.starts, epochs.stops)):
        ax.axvspan(
            start,
            stop,
            hatch=hatch,
            facecolor=fc,
            edgecolor=ec,
            alpha=alpha,
            label=label if ii == 0 else "_nolegend_",
            **kwargs,
        )

    if epochs.start < xmin:
        xmin = epochs.start
    if epochs.stop > xmax:
        xmax = epochs.stop
    ax.set_xlim([xmin, xmax])

    if hc is not None:
        try:
            mpl.rcParams["hatch.color"] = hc_before
        except UnboundLocalError:
            pass

    return ax
