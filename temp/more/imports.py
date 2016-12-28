import os
import sys
# import daft
import pickle
import numpy as np
import pandas as pd
import trackr as tkr # animal tracking library
import seaborn as sns
import eplotlib as epl # helper functions for creating figures
import matplotlib as mpl
import matplotlib.pyplot as plt

from IPython.display import display, clear_output

sys.path.insert(0, '../NeuroHMM')
sys.path.insert(0, '../NeuroHMM/helpers')

from efunctions import * # load my helper function(s) to save pdf figures, etc.
from hc3 import load_data, get_sessions
from hmmlearn import hmm # see https://github.com/ckemere/hmmlearn
import klabtools as klab
import seqtools as sq

import importlib

importlib.reload(sq) # reload module here only while prototyping...
importlib.reload(klab) # reload module here only while prototyping...

# %matplotlib inline

sns.set(style="ticks", context="paper",
        rc={"xtick.major.size": 3, "ytick.major.size": 3,
            "xtick.major.width": 1, "ytick.major.width": 1,
            "axes.linewidth": 1, "lines.linewidth": 1})
mpl.rcParams["savefig.dpi"] = 150
pd.set_option("display.precision", 3)

