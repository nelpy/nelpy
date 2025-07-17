# -*- coding: utf-8 -*-
"""
Description goes here.
"""

import os
import requests
import numpy as np

from collections import namedtuple

from tqdm import tqdm


def get_example_data_home(data_home=None):
    """Return the path of the nelpy example-data directory.
    This is used by the ``load_example_dataset`` function.
    If the ``data_home`` argument is not specified, the default location
    is ``~/nelpy-example-data``.
    Alternatively, a different default location can be specified using the
    environment variable ``NELPY_DATA``.
    """
    if data_home is None:
        data_home = os.environ.get(
            "NELPY_DATA", os.path.join("~", "nelpy-example-data")
        )
    data_home = os.path.expanduser(data_home)
    if not os.path.exists(data_home):
        os.makedirs(data_home)
    return data_home


def download_example_dataset(
    dataset_name=None, filename=None, data_home=None, overwrite=False, **kwargs
):
    """Download a dataset from the online repository (requires internet).

    Parameters
    ----------
    dataset_name : str, optional
        Name of the dataset (available at https://github.com/nelpy/example-data).
    filename : str, optional
        Filename to download from https://github.com/nelpy/example-data.
    data_home : string, optional
        The directory in which to cache data. By default, uses ~/nelpy-example-data/
    overwrite : boolean, optional
        If True, then existing data will be overwritten.
    kwargs : dict, optional
        Additional optional arguments.
    """
    filenames = []
    urls = []

    urlpath = "https://raw.githubusercontent.com/nelpy/example-data/master/{}"
    cachepath = os.path.join(get_example_data_home(data_home), "{}")

    # make sure that either name or file is specified, but not both
    if dataset_name is None:
        if filename is None:
            raise ValueError("either dataset_name or filename must be specified!")
    elif filename:
        raise ValueError("dataset_name and filename cannot both be specified!")

    if filename:
        filenames = [cachepath.format(filename)]
        urls = [urlpath.format(filename)]
    else:
        basenames = []
        if dataset_name == "linear-track":
            basenames.append("linear-track/trajectory.videoPositionTracking")
            basenames.append("linear-track/spikes.mat")

        elif dataset_name == "w-maze":
            basenames.append("w-maze/trajectory.videoPositionTracking")
            basenames.append("w-maze/spikes.mat")

        elif dataset_name == "spike-sorting":
            basenames.append("spike-sorting/spikedata.npz")

        elif dataset_name == "ripple-lfp":
            basenames.append("ripple-lfp/position.npz")
            basenames.append("ripple-lfp/tetrode2_1kHz.npz")
            basenames.append("ripple-lfp/tetrode3_1kHz.npz")

        elif dataset_name == "running-wheel":
            basenames.append("running-wheel/running-wheel.npz")
        else:
            raise ValueError(
                "example dataset_name '{}' not found!".format(dataset_name)
            )
        for basename in basenames:
            filenames.append(cachepath.format(basename))
            urls.append(urlpath.format(basename))

    downloaded_something = False
    for filename, url in zip(filenames, urls):
        if os.path.exists(filename) and not overwrite:
            print(
                "you already have {}, skipping download...".format(
                    os.path.basename(filename)
                )
            )
        else:
            downloaded_something = True
            print("downloading {}".format(url))
            datadir = os.path.dirname(filename)
            os.makedirs(datadir, exist_ok=True)

            # Streaming, so we can iterate over the response.
            r = requests.get(url, stream=True)

            # Total size in bytes.
            total_size = int(r.headers.get("content-length", 0))
            chunk_size = 1024  # number of bytes to process at a time (NOTE: progress bar unit only accurate if this is 1 kB)

            with open(filename, "wb+") as f:
                for data in tqdm(
                    r.iter_content(chunk_size),
                    total=int(total_size / chunk_size),
                    unit="kB",
                ):
                    f.write(data)
    if downloaded_something:
        print("data saved to local directory {}".format(filename))


def load_example_dataset(dataset_name=None, data_home=None, **kwargs):
    """Load an example dataset (may require internet).

    Parameters
    ----------
    dataset_name : str, optional
        Name of the dataset (available at https://github.com/nelpy/example-data).
    data_home : string, optional
        The directory in which to cache data. By default, uses ~/nelpy-example-data/
    kwargs : dict, optional
        Additional optional arguments.
    """

    cachepath = os.path.join(get_example_data_home(data_home), "{}")

    if dataset_name == "linear-track":
        raise NotImplementedError

    elif dataset_name == "w-maze":
        raise NotImplementedError

    elif dataset_name == "diba-linear":
        raise NotImplementedError

    elif dataset_name == "spike-sorting":
        filename = "spike-sorting/spikedata.npz"
        pathname = cachepath.format(filename)
        if not os.path.exists(pathname):
            print("file does not exist locally, attempting to download...")
            download_example_dataset(filename=filename, data_home=data_home)
        z = np.load(pathname)
        waveforms = z["waveforms"]
        spiketimes = z["spiketimes"]
        Data = namedtuple("Data", ["waveforms", "spiketimes"])
        data = Data(waveforms, spiketimes)

    elif dataset_name == "ripple-lfp":
        Data = namedtuple(
            "Data",
            [
                "T2timestamps",
                "T2data",
                "T3timestamps",
                "T3data",
                "posTimestamps",
                "posCoords",
            ],
        )
        filename = "ripple-lfp/position.npz"
        pathname = cachepath.format(filename)
        if not os.path.exists(pathname):
            print("file does not exist locally, attempting to download...")
            download_example_dataset(filename=filename, data_home=data_home)
        z = np.load(pathname)
        posTimestamps = z["timestamps"]
        posCoords = z["data"]

        filename = "ripple-lfp/tetrode2_1kHz.npz"
        pathname = cachepath.format(filename)
        if not os.path.exists(pathname):
            print("file does not exist locally, attempting to download...")
            download_example_dataset(filename=filename, data_home=data_home)
        z = np.load(pathname)
        T2timestamps = z["timestamps"]
        T2data = z["T2data"]

        filename = "ripple-lfp/tetrode3_1kHz.npz"
        pathname = cachepath.format(filename)
        if not os.path.exists(pathname):
            print("file does not exist locally, attempting to download...")
            download_example_dataset(filename=filename, data_home=data_home)
        z = np.load(pathname)
        T3timestamps = z["timestamps"]
        T3data = z["T3data"]

        data = Data(
            T2timestamps, T2data, T3timestamps, T3data, posTimestamps, posCoords
        )

    elif dataset_name == "running-wheel":
        filename = "running-wheel/running-wheel.npz"
        pathname = cachepath.format(filename)
        if not os.path.exists(pathname):
            print("file does not exist locally, attempting to download...")
            download_example_dataset(filename=filename, data_home=data_home)
        z = np.load(pathname)
        angle_to_linpos = z["angle_to_linpos"]
        angular_position = z["angular_position"]
        lick_lin_position = z["lick_lin_position"]
        lick_times = z["lick_times"]
        linear_position = z["linear_position"]
        motor_lin_position = z["motor_lin_position"]
        motor_times = z["motor_times"]
        reward_position = z["reward_position"]
        silent_position = z["silent_position"]
        timestamps = z["timestamps"]
        tracklength = z["tracklength"]
        wheel_avg_speed = z["wheel_avg_speed"]

        Data = namedtuple(
            "Data",
            [
                "angle_to_linpos",
                "angular_position",
                "lick_lin_position",
                "lick_times",
                "linear_position",
                "motor_lin_position",
                "motor_times",
                "reward_position",
                "silent_position",
                "timestamps",
                "tracklength",
                "wheel_avg_speed",
            ],
        )
        data = Data(
            angle_to_linpos,
            angular_position,
            lick_lin_position,
            lick_times,
            linear_position,
            motor_lin_position,
            motor_times,
            reward_position,
            silent_position,
            timestamps,
            tracklength,
            wheel_avg_speed,
        )
    else:
        raise ValueError("example dataset_name '{}' not found!".format(dataset_name))

    return data
