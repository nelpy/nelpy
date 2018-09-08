""" Data grouping objects """


"""
DataGroup
  |_

for example, CA1 + CA3 - interneurons

CA1.pyramidal
CA1.all
CA1 <==> CA1.all
CA1.interneurons

placecells.ctxA
placecells.ctxB
placecellsAB = placecells.ctxA & placecells.ctxB

placecellsAB.print()
placecellsAB.subgroups() --> returns e.g. pyramidal, all (builtin), interneurons

st[placecellsAB]
st[CA1.interneurons]

Possible example usages / extensions:

IMU.groups() --> {'mag': [0,1,2], 'acc': [3,4,5], 'gyro': [6,7,8]}
IMU.default() --> all

pos.groups() --> {'traj': [0,1], 'speed': 2}

"""