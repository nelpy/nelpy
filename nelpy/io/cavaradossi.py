"""A collection of functions used to extract data from Cavaradossi's
.dat files; Should be renamed sd or something.

x55 (Sync byte)
x68 (Data format byte - Note: MCU will replace this with Digital inputs)
1 byte with valid flags (bit3 = acc_valid, bit2 = gyro_valid, bit1 = mag_valid, bit0 = rf_valid) (Note: at most one bit set)
x00 (filler to make the packet byte count even)
i2c_data_x[7:0]   (or rf_timestamp[7:0] if rf_valid)
i2c_data_x[15:8]  (or rf_timestamp[15:8] if rf_valid)
i2c_data_y[7:0]   (or rf_timestamp[23:16] if rf_valid)
i2c_data_y[15:8]  (or rf_timestamp[31:24] if rf_valid)
i2c_data_z[7:0]   (or x00 if rf_valid)
i2c_data_z[15:8]  (or x00 if rf_valid)
timestamp[7:0]
timestamp[15:8]
timestamp[23:16]
timestamp[31:24]
for (ch = 0, ch < 32, ch++) {
  sample0[7:0]
  sample0[15:8]
  sample1[7:0]
  sample1[15:8]
  sample2[7:0]
  sample2[15:8]
  sample3[7:0]
  sample3[15:8]
}

Note: i2c_data[x,y,z] = 0 if no valid flag set.

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

    def read_sync_pulse(self, filename, duration=None):
        """duration in seconds."""

        # determine number of packets to extract
        num_packets = self.get_num_packets(filename=filename)
        if duration is None:
            max_packets = num_packets
        else:
            max_packets = duration*self.fs
        n_packets = np.min((max_packets, num_packets))

        # syncdata = np.zeros((n_packets, 2), dtype=np.uint32)
        # timestamps = np.zeros(n_packets)
        rf_syncs = []
        rf_timestamps = []

        ii = 0
        with open(filename, 'rb') as fileobj:
            for packet in iter(lambda: fileobj.read(self.packetSize), ''):
                if packet:
                    sp = struct.unpack('<I', packet[4:8])[0] # sync pulse
                    rf_invalid = struct.unpack('<H', packet[8:10])[0]
                    if not rf_invalid and sp:
                        rf_syncs.append(sp)
                        ts = struct.unpack('<I', packet[self.headerSize:self.headerSize+self.timestampSize])[0]
                        rf_timestamps.append(ts)
                    ii = ii+1
                else:
                    break
                if ii >= n_packets:
                    break
        asa = AnalogSignalArray(ydata=rf_syncs, timestamps=np.array(rf_timestamps)/self.fs, fs=self.fs)
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
