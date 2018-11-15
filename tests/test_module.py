import os
import random
import struct
import pytest

from rs_embed import EmbeddingData


CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
ID_PATH = os.path.join(CURRENT_DIR, '.test_ids.bin')
DATA_PATH = os.path.join(CURRENT_DIR, '.test_data.bin')

DIM = 128
N = 100000


@pytest.fixture(scope="session", autouse=True)
def dummy_data():
    with open(ID_PATH, "wb") as id_f, open(DATA_PATH, "wb") as data_f:
        for i in range(N):
            id_f.write((i.to_bytes(8, byteorder='little')))
            for _ in range(DIM):
                val = random.random() * 2 - 1.
                data_f.write(struct.pack('<f', val))
    yield
    os.remove(ID_PATH)
    os.remove(DATA_PATH)


def test_count():
    dataset = EmbeddingData(ID_PATH, DATA_PATH, DIM)
    assert dataset.count() == N


def test_get():
    dataset = EmbeddingData(ID_PATH, DATA_PATH, DIM)
    for i in range(N):
        v = dataset.get(i)
        assert len(v) == DIM


def test_nn_search():
    k = 1000
    dataset = EmbeddingData(ID_PATH, DATA_PATH, DIM)
    exemplar = [random.random() * 2 - 1. for i in range(N)]
    nn = dataset.nn([exemplar], k, float('inf'))
    assert len(nn) == k
    assert all(d[1] >= 0 for d in nn)
    assert all(nn[i][1] <= nn[i + 1][1] for i in range(len(nn) - 1))


def test_nn_search_by_id():
    k = 1000
    dataset = EmbeddingData(ID_PATH, DATA_PATH, DIM)

    nn = dataset.nn_by_id([0], k, float('inf'))
    assert len(nn) == k
    assert all(d[1] >= 0 for d in nn)
    assert all(nn[i][1] <= nn[i + 1][1] for i in range(len(nn) - 1))

    nn = dataset.nn_by_id(list(range(25)), k, float('inf'))
    assert len(nn) == k
    assert all(d[1] >= 0 for d in nn)
    assert all(nn[i][1] <= nn[i + 1][1] for i in range(len(nn) - 1))
