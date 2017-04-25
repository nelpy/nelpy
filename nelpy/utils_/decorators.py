import inspect

__all__ = ['add_method_to_instance',
           'add_method_to_class',
           'add_prop_to_instance',
           'add_prop_to_class']

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