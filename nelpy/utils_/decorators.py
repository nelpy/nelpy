import functools
import inspect
import warnings

__all__ = ['keyword_deprecation',
'           add_method_to_instance',
           'add_method_to_class',
           'add_prop_to_instance',
           'add_prop_to_class',
           'deprecated']

def keyword_deprecation(func=None, *, replace_x_with_y=None):
    """
    Keyword deprecator.

    Parameters
    ----------
    replace_x_with_y : dict
        Dictionary of kwargs to replace, {'old': 'new'}

    Example
    -------
    @keyword_deprecation(replace_x_with_y={'old1':'new1', 'old2':'new2'})
    def myfunc(arg1, arg2, *, new1=None, new2=5):
        pass
    """
    def _decorate(function):
        @functools.wraps(function)
        def wrapped_function(*args, **kwargs):
            if replace_x_with_y is not None:
                for oldkwarg, newkwarg in replace_x_with_y.items():
                    newvalue = kwargs.pop(newkwarg, None)
                    oldvalue = kwargs.pop(oldkwarg, None)
                    if newvalue is not None and oldvalue is not None:
                        raise ValueError("Cannot pass both '{}' and '{}'. Use '{}' instead.".format(oldkwarg, newkwarg, newkwarg))
                    if oldvalue is not None:
                        import warnings
                        warnings.warn("'{}' has been deprecated, use '{}' instead.".format(oldkwarg, newkwarg))
                        kwargs[newkwarg] = oldvalue
            return function(*args, **kwargs)
        return wrapped_function
    if func:
#         print('no args in decorator')
        return _decorate(func)

    return _decorate

def deprecated(func):
    '''This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used.'''

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.warn_explicit(
            "Call to deprecated function {}.".format(func.__name__),
            category=DeprecationWarning,
            filename=func.__code__.co_filename,
            lineno=func.__code__.co_firstlineno + 1
        )
        return func(*args, **kwargs)
    new_func.__name__ = func.__name__
    new_func.__doc__ = func.__doc__
    new_func.__dict__.update(func.__dict__)
    return new_func

# ## Usage examples ##
# @deprecated
# def my_func():
#     pass

# @other_decorators_must_be_upper
# @deprecated
# def my_func():
#     pass

def add_method_to_instance(instance):
    """Add a method to an object instance.

    Example
    -------

    >>> class Foo:
    >>> def __init__(self):
    >>>     self.x = 42

    >>> foo = Foo()

    >>> @add_method_to_instance(foo)
    >>> def print_x(self):
    >>>     \"""hello\"""
    >>>     print(self.x)

    """
    if inspect.isclass(instance):
        raise TypeError("instance expected, class object received")
    def decorator(f):
        import types
        f = types.MethodType(f, instance)
        setattr(instance, f.__name__, f)
        return f
    return decorator

def add_method_to_class(cls):
    """working for both class and instance inputs"""
    if not inspect.isclass(cls):
        cls = type(cls)
    def decorator(f):
        if not hasattr(cls, '__perinstance'):
            cls.__perinstance = True
        setattr(cls, f.__name__, f)
        return f
    return decorator

def add_prop_to_instance(instance):
    """working"""
    if inspect.isclass(instance):
        raise TypeError("instance expected, class object received")
    def decorator(f):
        cls = type(instance)
        cls = type(cls.__name__, (cls,), {})
        if not hasattr(cls, '__perinstance'):
            cls.__perinstance = True
        instance.__class__ = cls
        setattr(cls, f.__name__, property(f))
        return f
    return decorator

def add_prop_to_class(cls):
    """working"""
    if not inspect.isclass(cls):
        raise TypeError("class expected!")
    def decorator(f):
        if not hasattr(cls, '__perinstance'):
            cls.__perinstance = True
        setattr(cls, f.__name__, property(f))
        return f
    return decorator