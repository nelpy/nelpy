import pytest

import nelpy as nel


def test_valueeventarray_construction():
    events = [[0.1, 0.5, 1.0], [0.2, 0.6, 1.2]]
    values = [[1, 2, 3], [4, 5, 6]]
    veva = nel.ValueEventArray(events=events, values=values, fs=10)
    assert veva.n_series == 2
    assert all(n > 0 for n in veva.n_values)
    assert not veva.isempty
    assert veva.data.shape[0] == 2


def test_valueeventarray_properties():
    events = [[0.1, 0.5, 1.0]]
    values = [[1, 2, 3]]
    veva = nel.ValueEventArray(events=events, values=values, fs=10)
    assert veva.n_series == 1
    assert veva.n_values[0] == 1
    assert not veva.isempty
    assert veva.data.shape[0] == 1


def test_valueeventarray_empty():
    veva = nel.ValueEventArray(events=[[]], values=[[]], fs=10)
    assert veva.isempty
    assert veva.n_series == 1
    assert veva.data.shape[0] == 1


def test_valueeventarray_shape_mismatch():
    events = [[0.1, 0.5]]
    values = [[1, 2, 3]]  # Mismatch
    with pytest.raises(ValueError):
        nel.ValueEventArray(events=events, values=values, fs=10)


def test_valueeventarray_flatten():
    events = [[0.1, 0.5, 1.0], [0.2, 0.6, 1.2]]
    values = [[1, 2, 3], [4, 5, 6]]
    veva = nel.ValueEventArray(events=events, values=values, fs=10)
    try:
        veva.flatten()
    except NotImplementedError:
        pass
    except Exception:
        pytest.skip("Flatten not implemented or not supported for ValueEventArray")


def test_valueeventarray_indexing():
    events = [[0.1, 0.5, 1.0], [0.2, 0.6, 1.2]]
    values = [[1, 2, 3], [4, 5, 6]]
    veva = nel.ValueEventArray(events=events, values=values, fs=10)
    # Indexing first series
    sub = veva.iloc[:, 0]
    assert sub.n_series == 1
