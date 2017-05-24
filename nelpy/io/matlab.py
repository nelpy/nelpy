#encoding : utf-8
"""This module implements functionailty for loading Matlab .mat files"""

import scipy.io

def insert_into_namespace(name, value, name_space=None):
    if name_space is None:
        name_space = globals()
    name_space[name] = value

# mat = scipy.io.loadmat("simplified (Newton, 2015-03-11_15-09-22).mat", squeeze_me=True)
# vars = [key for key in mat.keys() if '__' not in key]
# varinfo = scipy.io.whosmat("simplified (Newton, 2015-03-11_15-09-22).mat", squeeze_me=True)

#NOTE: see https://github.com/frejanordsiek/hdf5storage for more comprehensive support

def load(filename, squeeze_me=True, name_space=None):
    """Load all variables from .mat file.

    NOTE: You will need an HDF5 python library to read matlab 7.3 or
        later format mat files.
    NOTE: To inject into global namespace, must be called with
        name_Space=globals()
    """

    mat = scipy.io.loadmat(filename, squeeze_me=squeeze_me)

    # optionally inject variables into desired name_space
    if name_space is not None:
        vars = [key for key in mat.keys() if '__' not in key]
        for var in vars:
            insert_into_namespace(var, mat[var], name_space)
            # Global.var = mat[var]

    return mat


# NOTE: You will need an HDF5 python library to read matlab 7.3 format mat files.
# Because scipy does not supply one, we do not implement the HDF5 / 7.3 interface here.