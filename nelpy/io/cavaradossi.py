"""A collection of functions used to extract data from Cavaradossi's .dat files"""

numChannels = 128
headerSize = 10
timestampSize = 4
channelSize = numChannels*2
packetSize = headerSize + timestampSize + channelSize


def bytes_from_file(filename, packetsize=270):
    with open(filename, "rb") as f:
        while True:
            packet = f.read(chunksize)
            if packet:
                for b in packet:
                    yield b
            else:
                break

# example:
for b in bytes_from_file('filename'):
    do_stuff_with(b)

# see http://stackoverflow.com/questions/22229229/reading-4-byte-integers-from-binary-file-in-python
import struct

timestamps = []
chdata = []
filename = 'sd7.dat'
ii = 0
with open(filename, 'rb') as fileobj:
    for packet in iter(lambda: fileobj.read(packetSize), ''):
        ii = ii+1
        if packet:
            ts = struct.unpack('<I', packet[headerSize:headerSize+timestampSize])[0]
            timestamps.append(ts)
            ch = struct.unpack('<h', packet[headerSize+timestampSize:headerSize+timestampSize+2])[0]
            chdata.append(ch)
            #integer_value = struct.unpack('<I', chunk)[0]
        else:
            break
        if ii > 1000000:
            break

plt.plot(np.array(timestamps),np.array(chdata))
plt.ylim([-1500, 1500])