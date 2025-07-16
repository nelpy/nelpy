"""Class definitions and utility functions specifically related to
experimental design and organization.
"""
import pandas as pd

class Trial:
    """
    Experimental trial container.

    This class is a placeholder for trial-specific metadata and methods.
    """
    pass


class Session:
    pass


class Cue:
    pass


class Shayok:
    pass


def combine_rats(data, rats, n_sessions, only_sound=False):
    """
    Combine behavioral measures from multiple rats, sessions, and trials.

    Parameters
    ----------
    data : dict
        Dictionary with rat (str) as key, containing Rat objects for each rat.
    rats : list
        List of rat IDs (str).
    n_sessions : int
        Number of sessions per rat.
    only_sound : bool, optional
        If True, only include trials with cue == 'sound'. Default is False.

    Returns
    -------
    df : pd.DataFrame
        DataFrame containing combined behavioral measures.
    """
    measures = ["durations", "numbers", "latency", "responses"]
    together = dict(
        trial=[],
        rat=[],
        session=[],
        trial_type=[],
        rewarded=[],
        cue=[],
        value=[],
        measure=[],
        condition=[],
    )

    for session in range(n_sessions):
        for rat in rats:
            for i, trial in enumerate(data[rat].sessions[session].trials):
                for measure in measures:
                    if not only_sound or trial.cue == "sound":
                        together["trial"].append("%s, %d" % (rat, i))
                        together["rat"].append(rat)
                        together["session"].append(session + 1)
                        together["trial_type"].append(trial.trial_type)
                        together["rewarded"].append(
                            "%s %s"
                            % (
                                trial.cue,
                                (
                                    "rewarded"
                                    if trial.trial_type % 2 == 0
                                    else "unrewarded"
                                ),
                            )
                        )
                        together["cue"].append(trial.cue)
                        together["condition"].append(
                            "%s %d" % (trial.cue, trial.trial_type)
                        )
                        together["measure"].append(measure)
                        # together["value"].append(f_analyze(trial, measure))

    df = pd.DataFrame(data=together)

    # fix_missing_trials(df)

    return df
