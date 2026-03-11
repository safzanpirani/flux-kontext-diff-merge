import sys
import types

import numpy as np


class FakeTensor:
    def __init__(self, array):
        self._array = np.asarray(array, dtype=np.float32)

    @property
    def shape(self):
        return self._array.shape

    def cpu(self):
        return self

    def detach(self):
        return self

    def numpy(self):
        return self._array

    def __getitem__(self, item):
        return FakeTensor(self._array[item])


def fake_from_numpy(array):
    return FakeTensor(np.array(array, copy=True))


def fake_cat(tensors, dim=0):
    arrays = [tensor.numpy() for tensor in tensors]
    return FakeTensor(np.concatenate(arrays, axis=dim))


sys.modules.setdefault(
    "torch",
    types.SimpleNamespace(
        from_numpy=fake_from_numpy,
        cat=fake_cat,
    ),
)
