"""
nelpy.analysis
=====

This is the nelpy analysis sub-package.

nelpy.analysis provides several commonly used analyses.
"""

# from .hmm_sparsity import HMMSurrogate
from .ergodic import fmpt, steady_state
from .replay import (
    column_cycle_array,
    get_significant_events,
    linregress_array,
    linregress_bst,
    linregress_ting,
    pooled_time_swap_bst,
    score_hmm_logprob_cumulative,
    score_hmm_time_resolved,
    three_consecutive_bins_above_q,
    time_swap_array,
    trajectory_score_array,
    trajectory_score_bst,
)

__all__ = [
    "linregress_ting",
    "linregress_array",
    "linregress_bst",
    "time_swap_array",
    "column_cycle_array",
    "trajectory_score_array",
    "trajectory_score_bst",
    "get_significant_events",
    "three_consecutive_bins_above_q",
    "score_hmm_time_resolved",
    "score_hmm_logprob_cumulative",
    "pooled_time_swap_bst",
    "steady_state",
    "fmpt",
]

__version__ = "0.0.1"  # should I maintain a separate version for this?
