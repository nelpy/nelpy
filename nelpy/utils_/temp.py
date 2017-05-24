
from functools import wraps

# return real part of a vector
def real_vector(vector):
  return map(lambda x: x.real, vector)

# return imaginary part of a vector
def imag_vector(vector):
  return map(lambda x: x.imag, vector)


class EpochSignalSlicer(object):
    def __init__(self, obj):
        self.obj = obj

    def __getitem__(self, *args):
        """epochs, signals"""
        try:
            slices = np.s_[args]; slices = slices[0]
        except TypeError:
            slices = np.s_[args]
        if len(slices) > 2:
            raise IndexError("only [epochs, signal] slicing is supported at this time!")
        elif len(slices) == 2:
            epochslice, signalslice = slices
        else:
            # only epochs to slice:
            epochslice = slices[0]
        return epochslice

lfp._ = EpochSignalSlicer(lfp)


def name_of_variable(var):
    """Return the name of an object"""
    for n,v in globals().items():
        print(n)
        if id(v) == id(var):
            return n
    return None


def tags(tag_name):
    def tags_decorator(func):
        @wraps(func)
        def func_wrapper(name):
            return "<{0}>{1}</{0}>".format(tag_name, func(name))
        return func_wrapper
    return tags_decorator

@tags("p")
def get_text(name):
    """returns some text"""
    return "Hello "+name

print get_text.__name__ # get_text
print get_text.__doc__ # returns some text
print get_text.__module__ # __main__

########################################################################

import warnings

def ignore_deprecation_warnings(func):
    '''This is a decorator which can be used to ignore deprecation warnings
    occurring in a function.'''
    def new_func(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            return func(*args, **kwargs)
    new_func.__name__ = func.__name__
    new_func.__doc__ = func.__doc__
    new_func.__dict__.update(func.__dict__)
    return new_func

# === Examples of use ===

@ignore_deprecation_warnings
def some_function_raising_deprecation_warning():
    warnings.warn("This is a deprecationg warning.",
                  category=DeprecationWarning)

class SomeClass:
    @ignore_deprecation_warnings
    def some_method_raising_deprecation_warning():
        warnings.warn("This is a deprecationg warning.",
                      category=DeprecationWarning)
########################################################################
'''
One of three degrees of enforcement may be specified by passing
the 'debug' keyword argument to the decorator:
    0 -- NONE:   No type-checking. Decorators disabled.
 #!python
-- MEDIUM: Print warning message to stderr. (Default)
    2 -- STRONG: Raise TypeError with message.
If 'debug' is not passed to the decorator, the default level is used.

Example usage:
    >>> NONE, MEDIUM, STRONG = 0, 1, 2
    >>>
    >>> @accepts(int, int, int)
    ... @returns(float)
    ... def average(x, y, z):
    ...     return (x + y + z) / 2
    ...
    >>> average(5.5, 10, 15.0)
    TypeWarning:  'average' method accepts (int, int, int), but was given
    (float, int, float)
    15.25
    >>> average(5, 10, 15)
    TypeWarning:  'average' method returns (float), but result is (int)
    15

Needed to cast params as floats in function def (or simply divide by 2.0).

    >>> TYPE_CHECK = STRONG
    >>> @accepts(int, debug=TYPE_CHECK)
    ... @returns(int, debug=TYPE_CHECK)
    ... def fib(n):
    ...     if n in (0, 1): return n
    ...     return fib(n-1) + fib(n-2)
    ...
    >>> fib(5.3)
    Traceback (most recent call last):
      ...
    TypeError: 'fib' method accepts (int), but was given (float)

'''
import sys

def accepts(*types, **kw):
    '''Function decorator. Checks decorated function's arguments are
    of the expected types.

    Parameters:
    types -- The expected types of the inputs to the decorated function.
             Must specify type for each parameter.
    kw    -- Optional specification of 'debug' level (this is the only valid
             keyword argument, no other should be given).
             debug = ( 0 | 1 | 2 )

    '''
    if not kw:
        # default level: MEDIUM
        debug = 1
    else:
        debug = kw['debug']
    try:
        def decorator(f):
            def newf(*args):
                if debug is 0:
                    return f(*args)
                assert len(args) == len(types)
                argtypes = tuple(map(type, args))
                if argtypes != types:
                    msg = info(f.__name__, types, argtypes, 0)
                    if debug is 1:
                        print >> sys.stderr, 'TypeWarning: ', msg
                    elif debug is 2:
                        raise TypeError, msg
                return f(*args)
            newf.__name__ = f.__name__
            return newf
        return decorator
    except KeyError, key:
        raise KeyError, key + "is not a valid keyword argument"
    except TypeError, msg:
        raise TypeError, msg


def returns(ret_type, **kw):
    '''Function decorator. Checks decorated function's return value
    is of the expected type.

    Parameters:
    ret_type -- The expected type of the decorated function's return value.
                Must specify type for each parameter.
    kw       -- Optional specification of 'debug' level (this is the only valid
                keyword argument, no other should be given).
                debug=(0 | 1 | 2)
    '''
    try:
        if not kw:
            # default level: MEDIUM
            debug = 1
        else:
            debug = kw['debug']
        def decorator(f):
            def newf(*args):
                result = f(*args)
                if debug is 0:
                    return result
                res_type = type(result)
                if res_type != ret_type:
                    msg = info(f.__name__, (ret_type,), (res_type,), 1)
                    if debug is 1:
                        print >> sys.stderr, 'TypeWarning: ', msg
                    elif debug is 2:
                        raise TypeError, msg
                return result
            newf.__name__ = f.__name__
            return newf
        return decorator
    except KeyError, key:
        raise KeyError, key + "is not a valid keyword argument"
    except TypeError, msg:
        raise TypeError, msg

def info(fname, expected, actual, flag):
    '''Convenience function returns nicely formatted error/warning msg.'''
    format = lambda types: ', '.join([str(t).split("'")[1] for t in types])
    expected, actual = format(expected), format(actual)
    msg = "'{}' method ".format( fname )\
          + ("accepts", "returns")[flag] + " ({}), but ".format(expected)\
          + ("was given", "result is")[flag] + " ({})".format(actual)
    return msg
########################################################################
from Queue import Queue
from threading import Thread

class asynchronous(object):
    def __init__(self, func):
        self.func = func

        def threaded(*args, **kwargs):
            self.queue.put(self.func(*args, **kwargs))

        self.threaded = threaded

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def start(self, *args, **kwargs):
        self.queue = Queue()
        thread = Thread(target=self.threaded, args=args, kwargs=kwargs);
        thread.start();
        return asynchronous.Result(self.queue, thread)

    class NotYetDoneException(Exception):
        def __init__(self, message):
            self.message = message

    class Result(object):
        def __init__(self, queue, thread):
            self.queue = queue
            self.thread = thread

        def is_done(self):
            return not self.thread.is_alive()

        def get_result(self):
            if not self.is_done():
                raise asynchronous.NotYetDoneException('the call has not yet completed its task')

            if not hasattr(self, 'result'):
                self.result = self.queue.get()

            return self.result

if __name__ == '__main__':
    # sample usage
    import time

    @asynchronous
    def long_process(num):
        time.sleep(10)
        return num * num

    result = long_process.start(12)

    for i in range(20):
        print i
        time.sleep(1)

        if result.is_done():
            print "result {0}".format(result.get_result())


    result2 = long_process.start(13)

    try:
        print "result2 {0}".format(result2.get_result())

    except asynchronous.NotYetDoneException as ex:
        print ex.message
########################################################################
import functools, logging


log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

class log_with(object):
    '''Logging decorator that allows you to log with a
specific logger.
'''
    # Customize these messages
    ENTRY_MESSAGE = 'Entering {}'
    EXIT_MESSAGE = 'Exiting {}'

    def __init__(self, logger=None):
        self.logger = logger

    def __call__(self, func):
        '''Returns a wrapper that wraps func.
The wrapper will log the entry and exit points of the function
with logging.INFO level.
'''
        # set logger if it was not set earlier
        if not self.logger:
            logging.basicConfig()
            self.logger = logging.getLogger(func.__module__)

        @functools.wraps(func)
        def wrapper(*args, **kwds):
            self.logger.info(self.ENTRY_MESSAGE.format(func.__name__))  # logging level .info(). Set to .debug() if you want to
            f_result = func(*args, **kwds)
            self.logger.info(self.EXIT_MESSAGE.format(func.__name__))   # logging level .info(). Set to .debug() if you want to
            return f_result
        return wrapper
########################################################################
# Sample use and output:

if __name__ == '__main__':
    logging.basicConfig()
    log = logging.getLogger('custom_log')
    log.setLevel(logging.DEBUG)
    log.info('ciao')

    @log_with(log)     # user specified logger
    def foo():
        print 'this is foo'
    foo()

    @log_with()        # using default logger
    def foo2():
        print 'this is foo2'
    foo2()
########################################################################
# output
>>> ================================ RESTART ================================
>>>
INFO:custom_log:ciao
INFO:custom_log:Entering foo # uses the correct logger
this is foo
INFO:custom_log:Exiting foo
INFO:__main__:Entering foo2  # uses the correct logger
this is foo2
INFO:__main__:Exiting foo2
########################################################################
import threading, sys, functools, traceback

def lazy_thunkify(f):
    """Make a function immediately return a function of no args which, when called,
    waits for the result, which will start being processed in another thread."""

    @functools.wraps(f)
    def lazy_thunked(*args, **kwargs):
        wait_event = threading.Event()

        result = [None]
        exc = [False, None]

        def worker_func():
            try:
                func_result = f(*args, **kwargs)
                result[0] = func_result
            except Exception, e:
                exc[0] = True
                exc[1] = sys.exc_info()
                print "Lazy thunk has thrown an exception (will be raised on thunk()):\n%s" % (
                    traceback.format_exc())
            finally:
                wait_event.set()

        def thunk():
            wait_event.wait()
            if exc[0]:
                raise exc[1][0], exc[1][1], exc[1][2]

            return result[0]

        threading.Thread(target=worker_func).start()

        return thunk

    return lazy_thunked
########################################################################

@lazy_thunkify
def slow_double(i):
    print "Multiplying..."
    time.sleep(5)
    print "Done multiplying!"
    return i*2


def maybe_multiply(x):
    double_thunk = slow_double(x)
    print "Thinking..."
    time.sleep(3)
    time.sleep(3)
    time.sleep(1)
    if x == 3:
        print "Using it!"
        res = double_thunk()
    else:
        print "Not using it."
        res = None
    return res

#both take 7 seconds
maybe_multiply(10)
maybe_multiply(3)