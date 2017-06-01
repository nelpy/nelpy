"""A collection of functions used to extract data from Cavaradossi's
.dat files; Should be renamed sd or something.
"""

"""
TODO: add info (duration of filename, prettyprint, bytes, etc.)
"""

import numpy as np
import struct
import os

from ..core import *

class SDReader():

    def __init__(self, numChannels=None, headerSize=None, timestampSize=None, channelSize=None, fs=None):
        if numChannels is None:
            numChannels = 128 # 128 channels (32 tetrodes)
        if headerSize is None:
            headerSize = 10 # 10 bytes header
        if timestampSize is None:
            timestampSize = 4 # 4 bytes timestamp
        if channelSize is None:
            channelSize = 2 # 2 bytes per EEG channel
        if fs is None:
            fs = 30000 # 30 kHz acquisition rate

        self.numChannels = numChannels
        self.headerSize = headerSize
        self.timestampSize = timestampSize
        self.channelSize = channelSize
        self.packetSize = headerSize + timestampSize + channelSize*numChannels
        self.fs = fs

    def get_num_packets(self, filename, packetSize=None):
        """Docstring goes here."""
        if packetSize is None:
            packetSize = self.packetSize
        num_packets = int(np.floor(os.path.getsize(filename)/packetSize))
        return num_packets

    def read_eeg(self, filename, channels=[0], duration=None, fs_out=None):
        """duration in seconds."""

        if fs_out is None:
            fs_out = self.fs

        # determine number of packets to extract
        num_packets = self.get_num_packets(filename=filename)
        if duration is None:
            max_packets = num_packets
        else:
            max_packets = duration*self.fs
        n_packets = np.min((max_packets, num_packets))

        n_channels = len(channels)
        chdata = np.zeros((n_channels, n_packets), dtype=np.float16)
        timestamps = np.zeros(n_packets)

        ii = 0
        with open(filename, 'rb') as fileobj:
            for packet in iter(lambda: fileobj.read(self.packetSize), ''):
                if packet:
                    ts = struct.unpack('<I', packet[self.headerSize:self.headerSize+self.timestampSize])[0]
                    timestamps[ii] = ts
                    for cc, channel in enumerate(channels):
                        ch = struct.unpack('<h', packet[self.headerSize+self.timestampSize+channel*2:self.headerSize+self.timestampSize+channel*2+2])[0]
                        chdata[cc,ii] = ch
                    #integer_value = struct.unpack('<I', chunk)[0]
                    ii = ii+1
                else:
                    break
                if ii >= n_packets:
                    break

        asa = AnalogSignalArray(chdata, timestamps=timestamps/self.fs, fs=self.fs)
        if fs_out != self.fs:
            asa = asa.subsample(fs=fs_out)

        return asa

# def bytes_from_file(filename, packetsize=270):
#     with open(filename, "rb") as f:
#         while True:
#             packet = f.read(chunksize)
#             if packet:
#                 for b in packet:
#                     yield b
#             else:
#                 break

# # example:
# for b in bytes_from_file('filename'):
#     do_stuff_with(b)

# see http://stackoverflow.com/questions/22229229/reading-4-byte-integers-from-binary-file-in-python

# timestamps = []
# chdata = []
# filename = 'sd7.dat'
# ii = 0
# with open(filename, 'rb') as fileobj:
#     for packet in iter(lambda: fileobj.read(packetSize), ''):
#         ii = ii+1
#         if packet:
#             ts = struct.unpack('<I', packet[headerSize:headerSize+timestampSize])[0]
#             timestamps.append(ts)
#             ch = struct.unpack('<h', packet[headerSize+timestampSize:headerSize+timestampSize+2])[0]
#             chdata.append(ch)
#             #integer_value = struct.unpack('<I', chunk)[0]
#         else:
#             break
#         if ii > 1000000:
#             break

# plt.plot(np.array(timestamps),np.array(chdata))
# plt.ylim([-1500, 1500])