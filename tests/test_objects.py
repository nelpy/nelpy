import pytest

import nelpy as nel

"""
Asserting proper raise exception:
=================================
# content of test_sysexit.py
import pytest
def f():
    raise SystemExit(1)

def test_mytest():
    with pytest.raises(SystemExit):
        f()
"""

# public static boolean nearlyEqual(float a, float b, float epsilon) {
# 		final float absA = Math.abs(a);
# 		final float absB = Math.abs(b);
# 		final float diff = Math.abs(a - b);

# 		if (a == b) { // shortcut, handles infinities
# 			return true;
# 		} else if (a == 0 || b == 0 || diff < Float.MIN_NORMAL) {
# 			// a or b is zero or both are extremely close to it
# 			// relative error is less meaningful here
# 			return diff < (epsilon * Float.MIN_NORMAL);
# 		} else { // use relative error
# 			return diff / Math.min((absA + absB), Float.MAX_VALUE) < epsilon;
# 		}
# 	}

class TestEpochArray:
    """Tests for EpochArray."""

    def emptyEpochArray(self):
        ep1 = nel.EpochArray([])
        assert ep1.isempty

    def duration2(self):
        ep1 = nel.EpochArray([[0,3],[7,10]], fs=2)
        # TODO: should be more careful and use a machine-equal test
        assert ep1.duration == 3

    def duration1(self):
        ep1 = nel.EpochArray([[0,3],[7,10]], fs=1)
        # TODO: should be more careful and use a machine-equal test
        assert ep1.duration == 3



# stdata = np.array([1,2,3,4,5,6,6.5,7,8,10])
# fs = 4
# st = SpikeTrain(stdata, fs=fs)
# st.cell_type = 'pyr'
# ep1 = EpochArray([])
# ep2 = EpochArray(np.array([[0,3],[7,10]]), fs=fs)
# ep3 = EpochArray(np.array([[0,3],[8,15]]), fs=fs)
# ep4 = EpochArray(np.array([[0,3],[17,20]]), fs=fs)
# ep5 = EpochArray(np.array([[17,20]]), fs=fs)
# ep6 = EpochArray(np.array([[-17,-10]]), fs=fs)
# print('SpikeTrain')
# print('==========')
# print(st, st.support)
# print('\nslicing with slice objects and integers')
# print('==========')
# print(st[:4], st[:4].support)
# print(st[4:], st[4:].support)
# print(st[4:50], st[4:50].support)
# print(st[40:], st[40:].support)
# print(st[-1], st[-1].support)
# print(st[80], st[80].support)
# print('\nslicing with EpochArrays')
# print('==========')
# print(st[ep1], st[ep1].support)
# print(st[ep2], st[ep2].support)
# print(st[ep3], st[ep3].support)
# print(st[ep4], st[ep4].support)
# print(st[ep5], st[ep5].support)
# print(st[ep6], st[ep6].support)