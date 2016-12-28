class Map(dict):
    """
    Example:
    m = Map({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
    """

    def __init__(self, *args, **kwargs):
        super(Map, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def __getattr__(self, attr):
        if attr == "__getnewargs_ex__":
            raise AttributeError("%r has no attribute %r" % (type(self), attr))
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(Map, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(Map, self).__delitem__(key)
        del self.__dict__[key]

    # TODO: extend class to include useful default functions and behaviors:
    # self.summary
    # self.notes
    # self.type
    # self.attributes

    # IMPORTANT!! This implementation has some issue when trying to create copies
    # see http://stackoverflow.com/questions/6891477/python-strange-error-typeerror-nonetype-object-is-not-callable
    # although I don't understand exactly where it fails yet... :( 
    # also see https://bugs.python.org/issue16251
