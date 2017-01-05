#encoding : utf-8
"""This file contains the nelpy plotting functions:
 * 
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from .objects import *

from matplotlib import cm
from matplotlib import colors as mplcolors

def matshow(data, *, ax=None, **kwargs):

    # Sort out default values for the parameters
    if ax is None:
        ax = plt.gca()

    # Handle different types of input data
    if isinstance(data, BinnedSpikeTrainArray):
        ax.matshow(data.data, **kwargs)
        warnings.warn("Automatic x-axis formatting not yet implemented")
    else:
        raise NotImplementedError(
            "matshow({}) not yet supported".format(str(type(data))))

    return ax

def plot(data, *, cmap=plt.cm.Accent, color=None, legend=True, ax=None, plot_support=True, **kwargs):
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

    # Handle different types of input data
    if isinstance(data, SpikeTrain):
        print("plotting SpikeTrain")
        raise NotImplementedError(
            "plotting {} not yet supported".format(str(type(data))))
    elif isinstance(data, SpikeTrainArray):
        colors = cmap(np.linspace(0.25, 0.75, data.n_units)) # TODO: if we go from 0 then most colormaps are invisible at one enf of the spectrum
        
        yrange = [.5, data.n_units + .5]
        
        for unit, spiketrain in enumerate(data.time):
            ax.vlines(spiketrain, unit + .55, unit + 1.45, colors=colors[unit], **kwargs)

        # change y-axis labels to integers
        yint = range(1, data.n_units+1)
        ax.set_yticks(yint)

        ax.set_ylim(yrange)

        # # plot the support epochs:
        # if plot_support:
        #     plot_epochs_hatch(ax=ax, epochs=data.support, height=data.n_units+1, fc='0.5', ec=None, alpha=0.2)

    elif isinstance(data, EpochArray):
        # TODO: how can I figure out an appropriate height when plotting a spiketrain support?
        plot_epochs_hatch(ax=ax, epochs=data, height=1e3, fc='0.2', ec=None, alpha=0.2, hatch=None)
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

def plot_epochs_hatch(ax, epochs, height=None, fc='0.5', ec=None,
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

# def tsplot(data, time=None, unit=None, condition=None, value=None,
#            err_style="ci_band", ci=68, interpolate=True, color=None,
#            estimator=np.mean, n_boot=5000, err_palette=None, err_kws=None,
#            legend=True, ax=None, **kwargs):
#     """Plot one or more timeseries with flexible representation of uncertainty.

#     This function is intended to be used with data where observations are
#     nested within sampling units that were measured at multiple timepoints.

#     It can take data specified either as a long-form (tidy) DataFrame or as an
#     ndarray with dimensions (unit, time) The interpretation of some of the
#     other parameters changes depending on the type of object passed as data.

#     Parameters
#     ----------
#     data : DataFrame or ndarray
#         Data for the plot. Should either be a "long form" dataframe or an
#         array with dimensions (unit, time, condition). In both cases, the
#         condition field/dimension is optional. The type of this argument
#         determines the interpretation of the next few parameters. When
#         using a DataFrame, the index has to be sequential.
#     time : string or series-like
#         Either the name of the field corresponding to time in the data
#         DataFrame or x values for a plot when data is an array. If a Series,
#         the name will be used to label the x axis.
#     unit : string
#         Field in the data DataFrame identifying the sampling unit (e.g.
#         subject, neuron, etc.). The error representation will collapse over
#         units at each time/condition observation. This has no role when data
#         is an array.
#     value : string
#         Either the name of the field corresponding to the data values in
#         the data DataFrame (i.e. the y coordinate) or a string that forms
#         the y axis label when data is an array.
#     condition : string or Series-like
#         Either the name of the field identifying the condition an observation
#         falls under in the data DataFrame, or a sequence of names with a length
#         equal to the size of the third dimension of data. There will be a
#         separate trace plotted for each condition. If condition is a Series
#         with a name attribute, the name will form the title for the plot
#         legend (unless legend is set to False).
#     err_style : string or list of strings or None
#         Names of ways to plot uncertainty across units from set of
#         {ci_band, ci_bars, boot_traces, boot_kde, unit_traces, unit_points}.
#         Can use one or more than one method.
#     ci : float or list of floats in [0, 100]
#         Confidence interval size(s). If a list, it will stack the error
#         plots for each confidence interval. Only relevant for error styles
#         with "ci" in the name.
#     interpolate : boolean
#         Whether to do a linear interpolation between each timepoint when
#         plotting. The value of this parameter also determines the marker
#         used for the main plot traces, unless marker is specified as a keyword
#         argument.
#     color : seaborn palette or matplotlib color name or dictionary
#         Palette or color for the main plots and error representation (unless
#         plotting by unit, which can be separately controlled with err_palette).
#         If a dictionary, should map condition name to color spec.
#     estimator : callable
#         Function to determine central tendency and to pass to bootstrap
#         must take an ``axis`` argument.
#     n_boot : int
#         Number of bootstrap iterations.
#     err_palette : seaborn palette
#         Palette name or list of colors used when plotting data for each unit.
#     err_kws : dict, optional
#         Keyword argument dictionary passed through to matplotlib function
#         generating the error plot,
#     legend : bool, optional
#         If ``True`` and there is a ``condition`` variable, add a legend to
#         the plot.
#     ax : axis object, optional
#         Plot in given axis; if None creates a new figure
#     kwargs :
#         Other keyword arguments are passed to main plot() call

#     Returns
#     -------
#     ax : matplotlib axis
#         axis with plot data

#     Examples
#     --------

#     Plot a trace with translucent confidence bands:

#     .. plot::
#         :context: close-figs

#         >>> import numpy as np; np.random.seed(22)
#         >>> import seaborn as sns; sns.set(color_codes=True)
#         >>> x = np.linspace(0, 15, 31)
#         >>> data = np.sin(x) + np.random.rand(10, 31) + np.random.randn(10, 1)
#         >>> ax = sns.tsplot(data=data)

#     Plot a long-form dataframe with several conditions:

#     .. plot::
#         :context: close-figs

#         >>> gammas = sns.load_dataset("gammas")
#         >>> ax = sns.tsplot(time="timepoint", value="BOLD signal",
#         ...                 unit="subject", condition="ROI",
#         ...                 data=gammas)

#     Use error bars at the positions of the observations:

#     .. plot::
#         :context: close-figs

#         >>> ax = sns.tsplot(data=data, err_style="ci_bars", color="g")

#     Don't interpolate between the observations:

#     .. plot::
#         :context: close-figs

#         >>> import matplotlib.pyplot as plt
#         >>> ax = sns.tsplot(data=data, err_style="ci_bars", interpolate=False)

#     Show multiple confidence bands:

#     .. plot::
#         :context: close-figs

#         >>> ax = sns.tsplot(data=data, ci=[68, 95], color="m")

#     Use a different estimator:

#     .. plot::
#         :context: close-figs

#         >>> ax = sns.tsplot(data=data, estimator=np.median)

#     Show each bootstrap resample:

#     .. plot::
#         :context: close-figs

#         >>> ax = sns.tsplot(data=data, err_style="boot_traces", n_boot=500)

#     Show the trace from each sampling unit:


#     .. plot::
#         :context: close-figs

#         >>> ax = sns.tsplot(data=data, err_style="unit_traces")

#     """
#     # Sort out default values for the parameters
#     if ax is None:
#         ax = plt.gca()

#     if err_kws is None:
#         err_kws = {}

#     # Handle different types of input data
#     if isinstance(data, pd.DataFrame):

#         xlabel = time
#         ylabel = value

#         # Condition is optional
#         if condition is None:
#             condition = pd.Series(np.ones(len(data)))
#             legend = False
#             legend_name = None
#             n_cond = 1
#         else:
#             legend = True and legend
#             legend_name = condition
#             n_cond = len(data[condition].unique())

#     else:
#         data = np.asarray(data)

#         # Data can be a timecourse from a single unit or
#         # several observations in one condition
#         if data.ndim == 1:
#             data = data[np.newaxis, :, np.newaxis]
#         elif data.ndim == 2:
#             data = data[:, :, np.newaxis]
#         n_unit, n_time, n_cond = data.shape

#         # Units are experimental observations. Maybe subjects, or neurons
#         if unit is None:
#             units = np.arange(n_unit)
#         unit = "unit"
#         units = np.repeat(units, n_time * n_cond)
#         ylabel = None

#         # Time forms the xaxis of the plot
#         if time is None:
#             times = np.arange(n_time)
#         else:
#             times = np.asarray(time)
#         xlabel = None
#         if hasattr(time, "name"):
#             xlabel = time.name
#         time = "time"
#         times = np.tile(np.repeat(times, n_cond), n_unit)

#         # Conditions split the timeseries plots
#         if condition is None:
#             conds = range(n_cond)
#             legend = False
#             if isinstance(color, dict):
#                 err = "Must have condition names if using color dict."
#                 raise ValueError(err)
#         else:
#             conds = np.asarray(condition)
#             legend = True and legend
#             if hasattr(condition, "name"):
#                 legend_name = condition.name
#             else:
#                 legend_name = None
#         condition = "cond"
#         conds = np.tile(conds, n_unit * n_time)

#         # Value forms the y value in the plot
#         if value is None:
#             ylabel = None
#         else:
#             ylabel = value
#         value = "value"

#         # Convert to long-form DataFrame
#         data = pd.DataFrame(dict(value=data.ravel(),
#                                  time=times,
#                                  unit=units,
#                                  cond=conds))

#     # Set up the err_style and ci arguments for the loop below
#     if isinstance(err_style, string_types):
#         err_style = [err_style]
#     elif err_style is None:
#         err_style = []
#     if not hasattr(ci, "__iter__"):
#         ci = [ci]

#     # Set up the color palette
#     if color is None:
#         current_palette = utils.get_color_cycle()
#         if len(current_palette) < n_cond:
#             colors = color_palette("husl", n_cond)
#         else:
#             colors = color_palette(n_colors=n_cond)
#     elif isinstance(color, dict):
#         colors = [color[c] for c in data[condition].unique()]
#     else:
#         try:
#             colors = color_palette(color, n_cond)
#         except ValueError:
#             color = mpl.colors.colorConverter.to_rgb(color)
#             colors = [color] * n_cond

#     # Do a groupby with condition and plot each trace
#     for c, (cond, df_c) in enumerate(data.groupby(condition, sort=False)):

#         df_c = df_c.pivot(unit, time, value)
#         x = df_c.columns.values.astype(np.float)

#         # Bootstrap the data for confidence intervals
#         boot_data = algo.bootstrap(df_c.values, n_boot=n_boot,
#                                    axis=0, func=estimator)
#         cis = [utils.ci(boot_data, v, axis=0) for v in ci]
#         central_data = estimator(df_c.values, axis=0)

#         # Get the color for this condition
#         color = colors[c]

#         # Use subroutines to plot the uncertainty
#         for style in err_style:

#             # Allow for null style (only plot central tendency)
#             if style is None:
#                 continue

#             # Grab the function from the global environment
#             try:
#                 plot_func = globals()["_plot_%s" % style]
#             except KeyError:
#                 raise ValueError("%s is not a valid err_style" % style)

#             # Possibly set up to plot each observation in a different color
#             if err_palette is not None and "unit" in style:
#                 orig_color = color
#                 color = color_palette(err_palette, len(df_c.values))

#             # Pass all parameters to the error plotter as keyword args
#             plot_kwargs = dict(ax=ax, x=x, data=df_c.values,
#                                boot_data=boot_data,
#                                central_data=central_data,
#                                color=color, err_kws=err_kws)

#             # Plot the error representation, possibly for multiple cis
#             for ci_i in cis:
#                 plot_kwargs["ci"] = ci_i
#                 plot_func(**plot_kwargs)

#             if err_palette is not None and "unit" in style:
#                 color = orig_color

#         # Plot the central trace
#         kwargs.setdefault("marker", "" if interpolate else "o")
#         ls = kwargs.pop("ls", "-" if interpolate else "")
#         kwargs.setdefault("linestyle", ls)
#         label = cond if legend else "_nolegend_"
#         ax.plot(x, central_data, color=color, label=label, **kwargs)

#     # Pad the sides of the plot only when not interpolating
#     ax.set_xlim(x.min(), x.max())
#     x_diff = x[1] - x[0]
#     if not interpolate:
#         ax.set_xlim(x.min() - x_diff, x.max() + x_diff)

#     # Add the plot labels
#     if xlabel is not None:
#         ax.set_xlabel(xlabel)
#     if ylabel is not None:
#         ax.set_ylabel(ylabel)
#     if legend:
#         ax.legend(loc=0, title=legend_name)

#     return ax

# # Subroutines for tsplot errorbar plotting
# # ----------------------------------------


# def _plot_ci_band(ax, x, ci, color, err_kws, **kwargs):
#     """Plot translucent error bands around the central tendancy."""
#     low, high = ci
#     if "alpha" not in err_kws:
#         err_kws["alpha"] = 0.2
#     ax.fill_between(x, low, high, facecolor=color, **err_kws)
