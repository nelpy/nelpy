#encoding : utf-8
"""This module implements functionailty for loading Matlab .mat files"""

import scipy.io
# import sys

# class global_injector:
#     '''Inject into the *real global namespace*, i.e. "builtins" namespace or "__builtin__" for python2.
#     Assigning to variables declared global in a function, injects them only into the module's global namespace.
#     >>> Global= sys.modules['__builtin__'].__dict__
#     >>> #would need 
#     >>> Global['aname'] = 'avalue'
#     >>> #With
#     >>> Global = global_injector()
#     >>> #one can do
#     >>> Global.bname = 'bvalue'
#     >>> #reading from it is simply
#     >>> bname
#     bvalue

#     '''
#     def __init__(self):
#         try:
#             self.__dict__['builtin'] = sys.modules['__builtin__'].__dict__
#         except KeyError:
#             self.__dict__['builtin'] = sys.modules['builtins'].__dict__
#     def __setattr__(self,name,value):
#         self.builtin[name] = value

def insert_into_namespace(name, value, name_space=globals()):
    name_space[name] = value

# mat = scipy.io.loadmat("simplified (Newton, 2015-03-11_15-09-22).mat", squeeze_me=True)
# vars = [key for key in mat.keys() if '__' not in key]
# varinfo = scipy.io.whosmat("simplified (Newton, 2015-03-11_15-09-22).mat", squeeze_me=True)

def load_all_from_mat(filename, name_space):
    """Load all variables from .mat file.

    NOTE: must be called with name_Space=globals()
    """

    mat = scipy.io.loadmat(filename, squeeze_me=True)
    vars = [key for key in mat.keys() if '__' not in key]

    for var in vars:
        insert_into_namespace(var, mat[var], name_space)
        # Global.var = mat[var]


# NOTE: You will need an HDF5 python library to read matlab 7.3 format mat files.
# Because scipy does not supply one, we do not implement the HDF5 / 7.3 interface here.