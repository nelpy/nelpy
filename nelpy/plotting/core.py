from ..objects import *

def imshow(data, *, ax=None, interpolation=None, **kwargs):
    """Docstring goes here."""

    # set default interpolation mode to 'none'
    if interpolation is None:
        interpolation = 'none'

def matshow(data, *, ax=None, **kwargs):
    """Docstring goes here."""

    # Sort out default values for the parameters
    if ax is None:
        ax = plt.gca()

    # Handle different types of input data
    if isinstance(data, BinnedSpikeTrainArray):
        # TODO: split by epoch, and plot matshows in same row, but with
        # a small gap to indicate discontinuities. How about slicing
        # then? Or slicing within an epoch?
        ax.matshow(data.data, **kwargs)
        ax.set_xlabel('time')
        ax.set_ylabel('unit')
        warnings.warn("Automatic x-axis formatting not yet implemented")
    else:
        raise NotImplementedError(
            "matshow({}) not yet supported".format(str(type(data))))

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
    pass

def overviewstrip():
    """Plot an epoch array similar to vs scrollbar, to show gaps in e.g.
    matshow plots.
    """
    pass


def plot(data, *, cmap=None, color=None, legend=True, ax=None, plot_support=True, **kwargs):
    """Plot one or more timeseries with flexible representation of uncertainty.

    This function is intended to be used with data where observations are
    nested within sampling units that were measured at multiple timepoints.

    It can take data specified either as a long-form (tidy) DataFrame or as an
    ndarray with dimensions (unit, time) The interpretation of some of the
    other parameters changes depending on the type of object passed as data.

    Parameters
    ----------
    data : DataFrame or ndarray
        Data for the plot. Should either be a "long form" dataframe or an
        array with dimensions (unit, time, condition). In both cases, the
        condition field/dimension is optional. The type of this argument
        determines the interpretation of the next few parameters. When
        using a DataFrame, the index has to be sequential.
    time : string or series-like
        Either the name of the field corresponding to time in the data
        DataFrame or x values for a plot when data is an array. If a Series,
        the name will be used to label the x axis.
    unit : string
        Field in the data DataFrame identifying the sampling unit (e.g.
        subject, neuron, etc.). The error representation will collapse over
        units at each time/condition observation. This has no role when data
        is an array.
    value : string
        Either the name of the field corresponding to the data values in
        the data DataFrame (i.e. the y coordinate) or a string that forms
        the y axis label when data is an array.
    condition : string or Series-like
        Either the name of the field identifying the condition an observation
        falls under in the data DataFrame, or a sequence of names with a length
        equal to the size of the third dimension of data. There will be a
        separate trace plotted for each condition. If condition is a Series
        with a name attribute, the name will form the title for the plot
        legend (unless legend is set to False).
    err_style : string or list of strings or None
        Names of ways to plot uncertainty across units from set of
        {ci_band, ci_bars, boot_traces, boot_kde, unit_traces, unit_points}.
        Can use one or more than one method.
    ci : float or list of floats in [0, 100]
        Confidence interval size(s). If a list, it will stack the error
        plots for each confidence interval. Only relevant for error styles
        with "ci" in the name.
    interpolate : boolean
        Whether to do a linear interpolation between each timepoint when
        plotting. The value of this parameter also determines the marker
        used for the main plot traces, unless marker is specified as a keyword
        argument.
    color : seaborn palette or matplotlib color name or dictionary
        Palette or color for the main plots and error representation (unless
        plotting by unit, which can be separately controlled with err_palette).
        If a dictionary, should map condition name to color spec.
    estimator : callable
        Function to determine central tendency and to pass to bootstrap
        must take an ``axis`` argument.
    n_boot : int
        Number of bootstrap iterations.
    err_palette : seaborn palette
        Palette name or list of colors used when plotting data for each unit.
    err_kws : dict, optional
        Keyword argument dictionary passed through to matplotlib function
        generating the error plot,
    legend : bool, optional
        If ``True`` and there is a ``condition`` variable, add a legend to
        the plot.
    ax : axis object, optional
        Plot in given axis; if None creates a new figure
    kwargs :
        Other keyword arguments are passed to main plot() call

    Returns
    -------
    ax : matplotlib axis
        axis with plot data

    Examples
    --------

    Plot a trace with translucent confidence bands:

    .. plot::
        :context: close-figs

        >>> import numpy as np; np.random.seed(22)
        >>> import seaborn as sns; sns.set(color_codes=True)
        >>> x = np.linspace(0, 15, 31)
        >>> data = np.sin(x) + np.random.rand(10, 31) + np.random.randn(10, 1)
        >>> ax = sns.tsplot(data=data)

    Plot a long-form dataframe with several conditions:

    .. plot::
        :context: close-figs

        >>> gammas = sns.load_dataset("gammas")
        >>> ax = sns.tsplot(time="timepoint", value="BOLD signal",
        ...                 unit="subject", condition="ROI",
        ...                 data=gammas)

    Use error bars at the positions of the observations:

    .. plot::
        :context: close-figs

        >>> ax = sns.tsplot(data=data, err_style="ci_bars", color="g")

    Don't interpolate between the observations:

    .. plot::
        :context: close-figs

        >>> import matplotlib.pyplot as plt
        >>> ax = sns.tsplot(data=data, err_style="ci_bars", interpolate=False)

    Show multiple confidence bands:

    .. plot::
        :context: close-figs

        >>> ax = sns.tsplot(data=data, ci=[68, 95], color="m")

    Use a different estimator:

    .. plot::
        :context: close-figs

        >>> ax = sns.tsplot(data=data, estimator=np.median)

    Show each bootstrap resample:

    .. plot::
        :context: close-figs

        >>> ax = sns.tsplot(data=data, err_style="boot_traces", n_boot=500)

    Show the trace from each sampling unit:


    .. plot::
        :context: close-figs

        >>> ax = sns.tsplot(data=data, err_style="unit_traces")

    """
    # TODO: change y-axis formatter to be only integers, and add default
    # labels

    # Sort out default values for the parameters
    if ax is None:
        ax = plt.gca()
    if cmap is None and color is None:
        color = 'k'

    # Handle different types of input data
    if isinstance(data, SpikeTrain):
        print("plotting SpikeTrain")
        raise NotImplementedError(
            "plotting {} not yet supported".format(str(type(data))))
    elif isinstance(data, SpikeTrainArray):
        yrange = [.5, data.n_units + .5]

        if cmap is not None:
            colors = cmap(np.linspace(0.25, 0.75, data.n_units)) # TODO: if we go from 0 then most colormaps are invisible at one end of the spectrum
            for unit, spiketrain in enumerate(data.time):
                ax.vlines(spiketrain, unit + .55, unit + 1.45, colors=colors[unit], **kwargs)
        else:  # use a constant color:
            for unit, spiketrain in enumerate(data.time):
                ax.vlines(spiketrain, unit + .55, unit + 1.45, colors=color, **kwargs)

        # change y-axis labels to integers
        yint = range(1, data.n_units+1)
        ax.set_yticks(yint)

        ax.set_ylim(yrange)

        # # plot the support epochs:
        # if plot_support:
        #     epoch_plot(ax=ax, epochs=data.support, height=data.n_units+1, fc='0.5', ec=None, alpha=0.2)

    elif isinstance(data, EpochArray):
        # TODO: how can I figure out an appropriate height when plotting a spiketrain support?
        epoch_plot(ax=ax, epochs=data, height=1e3, fc='0.2', ec=None, alpha=0.2, hatch=None)
    else:
        raise NotImplementedError(
            "plotting {} not yet supported".format(str(type(data))))

    # ax.plot(x, central_data, color=color, label=label, **kwargs)

    # # Pad the sides of the plot only when not interpolating
    # ax.set_xlim(x.min(), x.max())
    # x_diff = x[1] - x[0]
    # if not interpolate:
    #     ax.set_xlim(x.min() - x_diff, x.max() + x_diff)

    # # Add the plot labels
    # if xlabel is not None:
    #     ax.set_xlabel(xlabel)
    # if ylabel is not None:
    #     ax.set_ylabel(ylabel)
    # if legend:
    #     ax.legend(loc=0, title=legend_name)

    return ax

def epoch_plot(ax, epochs, height=None, fc='0.5', ec=None,
                      alpha=0.5, hatch='////'):
    """Docstring goes here.
    """

    import matplotlib.patches as patches
    for start, stop in zip(epochs.starts, epochs.stops):
        ax.add_patch(
            patches.Rectangle(
                (start, 0),   # (x,y)
                stop - start ,          # width
                height,          # height
                hatch=hatch,
                facecolor=fc,
                edgecolor=ec,
                alpha=alpha
            )
        )
    # ax.set_xlim([epochs.start, epochs.stop])