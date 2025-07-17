"""Results container object to easily group, store, and load results."""

__all__ = ["ResultsContainer", "load_pkl", "save_pkl"]

import gzip
import os

import dill as pickle

# import inspect


class ResultsContainer(object):
    """
    Extremely simple namespace for passing around and pickling data.

    This container is used to group, store, and load results from analyses or experiments.

    Parameters
    ----------
    description : str, optional
        Description of the results container.
    **kwargs :
        Additional attributes to store in the container.

    Attributes
    ----------
    description : str
        Description of the results container.
    _index : int
        Internal index for iteration.
    """

    def __init__(self, *args, description=None, **kwargs):
        """
        Initialize a ResultsContainer.

        Parameters
        ----------
        description : str, optional
            Description of the results container.
        **kwargs :
            Additional attributes to store in the container.
        """
        kwargs["description"] = description

        if len(args) > 0:
            raise NotImplementedError("only keyword arguments accepted!")

        for key, val in kwargs.items():
            setattr(self, key, val)

        self._index = 0

    # def __init__(self, *args, description=None, **kwargs):
    #     kwargs['description'] = description

    #     # BEGIN very hacky code to get *args names; might break!
    #     # see http://stackoverflow.com/questions/2749796/how-to-get-the-original-variable-name-of-variable-passed-to-a-function
    #     frame = inspect.currentframe()
    #     frame = inspect.getouterframes(frame)[1]
    #     string = inspect.getframeinfo(frame[0]).code_context[0].strip()
    #     _args = string[string.find('(') + 1:-1].split(',')

    #     names = []
    #     for i in _args:
    #         if i.find('=') != -1:
    #             pass
    #             # names.append(i.split('=')[1].strip())
    #         else:
    #             names.append(i)

    #     for ii, arg in enumerate(args):
    #         setattr(self, names[ii], arg)
    #     # END very hacky code to get *args names; might break!

    #     for key, val in kwargs.items():
    #         setattr(self, key, val)

    #     self.description = None

    def __repr__(self):
        """
        Return a string representation of the ResultsContainer.

        Returns
        -------
        repr_str : str
            String representation of the ResultsContainer, including address, number of objects, and description.
        """
        if self.isempty:
            return "<Empty ResultsContainer>"
        if self.n_objects == 1:
            n_str = " (" + str(self.n_objects) + " object)"
        else:
            n_str = " (" + str(self.n_objects) + " objects)"
        address_str = " at " + str(hex(id(self)))
        descr_str = ""
        if self.description is not None:
            descr_str = "\n**Description:** " + str(self.description)
        return "<ResultsContainer" + address_str + n_str + ">" + descr_str

    # def __iter__(self):
    #     self._index = 0
    #     return self

    # def __next__(self):
    #     index = self._index

    #     if index > self.n_objects - 1:
    #         raise StopIteration
    #     self._index += 1
    #     return self[index]

    # def __getitem__(self, idx):
    #     keys = [key for key in self.__dict__ if key not in ['description', '_index']]
    #     keys.sort()
    #     return keys[idx]

    @property
    def isempty(self):
        """Empty ResultsContainer."""
        return self.n_objects == 0

    @property
    def n_objects(self):
        """(int) Number of objects in ResultsContainer."""
        return len(list(self.__dict__.values())) - 2

    @property
    def keys(self):
        """(list) Key names in ResultsContainer."""
        keys = [key for key in self.__dict__ if key not in ["description", "_index"]]
        keys.sort()
        return keys

    def save_pkl(self, fname, zip=True, overwrite=False):
        """Write pickled data to disk, possible compressing."""
        if os.path.isfile(fname):
            # file exists
            if overwrite:
                pass
            else:
                print('File "{}" already exists! Aborting...'.format(fname))
                return
        if zip:
            save_large_file_without_zip = False
            with gzip.open(fname, "wb") as fid:
                try:
                    pickle.dump(self, fid)
                except OverflowError:
                    print(
                        "writing to disk using protocol=4, which supports file sizes > 4 GiB, and ignoring zip=True (zip is not supported for large files yet)"
                    )
                    save_large_file_without_zip = True

            if save_large_file_without_zip:
                with open(fname, "wb") as fid:
                    pickle.dump(self, fid, protocol=4)
        else:
            with open(fname, "wb") as fid:
                try:
                    pickle.dump(self, fid)
                except OverflowError:
                    print(
                        "writing to disk using protocol=4, which supports file sizes > 4 GiB"
                    )
                    pickle.dump(self, fid, protocol=4)


def load_pkl(fname, zip=True):
    """Read pickled data from disk, possible decompressing."""
    if zip:
        try:
            with gzip.open(fname, "rb") as fid:
                res = pickle.load(fid)
        except OSError:
            # most likely, results were not saved using zip=True, so let's try
            # to load without zip:
            with open(fname, "rb") as fid:
                res = pickle.load(fid)
    else:
        with open(fname, "rb") as fid:
            res = pickle.load(fid)
    return res


def save_pkl(fname, res, zip=True, overwrite=False):
    """Write pickled data to disk, possible compressing."""
    if os.path.isfile(fname):
        # file exists
        if overwrite:
            pass
        else:
            print('File "{}" already exists! Aborting...'.format(fname))
            return
    if zip:
        save_large_file_without_zip = False
        with gzip.open(fname, "wb") as fid:
            try:
                pickle.dump(res, fid)
            except OverflowError:
                print(
                    "writing to disk using protocol=4, which supports file sizes > 4 GiB, and ignoring zip=True (zip is not supported for large files yet)"
                )
                save_large_file_without_zip = True

        if save_large_file_without_zip:
            with open(fname, "wb") as fid:
                pickle.dump(res, fid, protocol=4)
    else:
        with open(fname, "wb") as fid:
            try:
                pickle.dump(res, fid)
            except OverflowError:
                print(
                    "writing to disk using protocol=4, which supports file sizes > 4 GiB"
                )
                pickle.dump(res, fid, protocol=4)
