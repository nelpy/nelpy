import functools
import inspect
import warnings

__all__ = ['add_method_to_instance',
           'add_method_to_class',
           'add_prop_to_instance',
           'add_prop_to_class',
           'deprecated']

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