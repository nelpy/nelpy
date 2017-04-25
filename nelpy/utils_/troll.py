import getpass
import inspect

from .. import core
# from .decorators import *

def add_prop_to_class(cls, name):
    """working"""
    if not inspect.isclass(cls):
        raise TypeError("class expected!")
    def decorator(f):
        if not hasattr(cls, '__perinstance'):
            cls.__perinstance = True
        setattr(cls, name, property(f))
        return f
    return decorator

@add_prop_to_class(core.AnalogSignalArray, getpass.getuser())
def whos_your_daddy(self):
    return getpass.getuser()*10000

