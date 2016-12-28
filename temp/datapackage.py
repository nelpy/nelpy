from path import path
import hashlib
import json
import numpy as np
import os
import pandas as pd
from datetime import datetime


def md5(data_path):
    # we need to compute the md5 sum one chunk at a time, because some
    # files are too large to fit in memory
    md5 = hashlib.md5()
    with open(data_path, 'r') as fh:
        while True:
            chunk = fh.read(128)
            if not chunk:
                break
            md5.update(chunk)
    data_hash = md5.hexdigest()
    return data_hash


class DataPackage(dict):

    def __init__(self, name, licenses):
        self['name'] = name
        self['datapackage_version'] = '1.0-beta.5'
        self['licenses'] = []
        for l in licenses:
            if not isinstance(l, dict):
                if l == 'odc-by':
                    url = 'http://opendefinition.org/licenses/odc-by'
                else:
                    raise ValueError("unrecognized license: %s" % l)
                l = dict(id=l, url=url)
            self['licenses'].append(l)

        self['title'] = None
        self['description'] = None
        self['homepage'] = None
        self['version'] = '0.0.1'
        self['sources'] = []
        self['keywords'] = None
        self['last_modified'] = datetime.now().isoformat(" ")
        self['image'] = None
        self['contributors'] = []
        self['resources'] = []

        self._path = None
        self._resource_map = {}

    @property
    def abspath(self):
        return self._path.joinpath(self['name']).abspath()

    @classmethod
    def load(cls, pth):
        pth = path(pth)

        dpjson_pth = pth.joinpath("datapackage.json")
        if not dpjson_pth.exists():
            raise IOError("No metadata file datapackage.json")
        with open(dpjson_pth, "r") as fh:
            dpjson = json.load(fh)

        name = dpjson['name']
        licenses = dpjson['licenses']
        resources = dpjson['resources']
        del dpjson['name']
        del dpjson['licenses']
        del dpjson['resources']

        dp = cls(name=name, licenses=licenses)
        dp._path = pth.splitpath()[0]
        dp.update(dpjson)

        if dp.abspath != pth.abspath():
            raise ValueError("malformed datapackage")

        for resource in resources:
            rname = resource['name']
            rfmt = resource['format']
            rpth = resource.get('path', None)
            rdata = resource.get('data', None)

            del resource['name']
            del resource['format']
            if 'path' in resource:
                del resource['path']
            if 'data' in resource:
                del resource['data']

            r = Resource(name=rname, fmt=rfmt, pth=rpth, data=rdata)
            r.update(resource)
            dp.add_resource(r)

        return dp

    def add_contributor(self, name, email):
        self['contributors'].append(dict(name=name, email=email))

    def clear_resources(self):
        self['resources'] = []
        self._resource_map = {}

    def add_resource(self, resource):
        self['resources'].append(resource)
        self._resource_map[resource['name']] = len(self['resources']) - 1
        resource.dpkg = self

    def get_resource(self, name):
        return self['resources'][self._resource_map[name]]

    def load_resource(self, name, verify=True):
        return self.get_resource(name).load_data(verify=verify)

    def load_resources(self, verify=True):
        for resource in self['resources']:
            resource.load_data(verify=verify)

    def save_metadata(self, dest=None):
        if dest:
            self._path = dest
        self['last_modified'] = datetime.now().isoformat(" ")
        metapath = self.abspath.joinpath("datapackage.json")
        with open(metapath, "w") as fh:
            json.dump(self, fh, indent=2)

    def save_data(self, dest=None):
        if dest:
            self._path = dest
        for resource in self['resources']:
            resource.save_data()

    def save(self, dest=None):
        if dest:
            self._path = dest
        if not self.abspath.exists():
            self.abspath.makedirs_p()
        self.save_data()
        self.save_metadata()

    def bump_major_version(self):
        major, minor, patch = map(int, self['version'].split("."))
        major += 1
        minor = 0
        patch = 0
        self['version'] = "%d.%d.%d" % (major, minor, patch)
        return self['version']

    def bump_minor_version(self):
        major, minor, patch = map(int, self['version'].split("."))
        minor += 1
        patch = 0
        self['version'] = "%d.%d.%d" % (major, minor, patch)
        return self['version']

    def bump_patch_version(self):
        major, minor, patch = map(int, self['version'].split("."))
        patch += 1
        self['version'] = "%d.%d.%d" % (major, minor, patch)
        return self['version']


class Resource(dict):

    def __init__(self, name, fmt, data=None, pth=None):
        self['name'] = name
        self['modified'] = datetime.now().isoformat(" ")
        self['format'] = fmt

        if pth:
            self['path'] = pth

        self.data = data
        self.dpkg = None

    @property
    def abspath(self):
        if not self.get('path', None):
            raise ValueError("no relative path specified")
        if not self.dpkg:
            raise ValueError("no linked datapackage")
        return self.dpkg.abspath.joinpath(self['path'])

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, val):
        self._data = val
        if 'path' not in self:
            self['data'] = val

    def save_data(self):
        self['modified'] = datetime.now().isoformat(" ")

        if 'path' not in self:
            return

        if self['format'] == 'csv':
            pd.DataFrame(self.data).to_csv(self.abspath)
        elif self['format'] == 'json':
            with open(self.abspath, "w") as fh:
                json.dump(self.data, fh)
        elif self['format'] == 'npy':
            np.save(self.abspath, np.array(self.data))
        else:
            raise ValueError("unsupported format: %s" % self['format'])

        self.update_size()
        self.update_hash()

    def load_data(self, verify=True):
        if self.data is not None:
            return self.data

        # check the file size
        if self.update_size():
            raise IOError("resource has changed size on disk")

        # load the raw data and check md5
        if verify and self.update_hash():
            raise IOError("resource checksum has changed")

        # check format and load data
        if self['format'] == 'csv':
            data = pd.DataFrame.from_csv(self.abspath)
        elif self['format'] == 'json':
            with open(self.abspath, "r") as fh:
                data = json.load(fh)
        elif self['format'] == 'npy':
            data = np.load(self.abspath, mmap_mode='c')
        else:
            raise ValueError("unsupported format: %s" % self['format'])

        self.data = data
        return self.data

    def update_size(self):
        old_size = self.get('bytes', None)
        new_size = self.abspath.getsize()
        self['bytes'] = new_size
        return old_size != new_size

    def update_hash(self):
        old_hash = self.get('hash', None)
        new_hash = md5(self.abspath)
        self['hash'] = new_hash
        return old_hash != new_hash
