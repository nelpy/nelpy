"""
nelpy.analysis
=====

This is the nelpy analysis sub-package.

nelpy.analysis provides several commonly used analyses.
"""

# from .hmm_sparsity import HMMSurrogate
from .replay import (
    linregress_ting,
    linregress_array,
    linregress_bst,
    time_swap_array,
    column_cycle_array,
    trajectory_score_array,
    trajectory_score_bst,
    get_significant_events,
    three_consecutive_bins_above_q,
    score_hmm_time_resolved,
    score_hmm_logprob_cumulative,
    pooled_time_swap_bst,
)
from .ergodic import steady_state, fmpt


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
