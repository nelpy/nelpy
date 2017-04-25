import getpass
from .. import core
from .decorators import *

@add_prop_to_class(core.AnalogSignalArray)
def whos_your_daddy(self):
    return getpass.getuser()

