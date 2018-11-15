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
    emb_data = EmbeddingData(ID_PATH, DATA_PATH, DIM)
    assert emb_data.count() == N


def test_get():
    emb_data = EmbeddingData(ID_PATH, DATA_PATH, DIM)
    for i in range(N):
        v = emb_data.get(i)
        assert len(v) == DIM


def test_nn_search():
    k = 1000
    emb_data = EmbeddingData(ID_PATH, DATA_PATH, DIM)
    exemplar = [random.random() * 2 - 1. for i in range(N)]
    nn = emb_data.nn([exemplar], k, float('inf'))
    assert len(nn) == k
    assert all(d[1] >= 0 for d in nn)
    assert all(nn[i][1] <= nn[i + 1][1] for i in range(len(nn) - 1))


def test_nn_search_by_id():
    k = 1000
    emb_data = EmbeddingData(ID_PATH, DATA_PATH, DIM)

    nn = emb_data.nn_by_id([0], k, float('inf'))
    assert len(nn) == k
    assert all(d[1] >= 0 for d in nn)
    assert all(nn[i][1] <= nn[i + 1][1] for i in range(len(nn) - 1))

    nn = emb_data.nn_by_id(list(range(25)), k, float('inf'))
    assert len(nn) == k
    assert all(d[1] >= 0 for d in nn)
    assert all(nn[i][1] <= nn[i + 1][1] for i in range(len(nn) - 1))


def test_kmeans():
    k = 10
    emb_data = EmbeddingData(ID_PATH, DATA_PATH, DIM)
    clusters = {}
    for i, c in emb_data.kmeans(list(range(N)), k):
        if c not in clusters:
            clusters[c] = []
        clusters[c].append(i)
    assert len(clusters) == k
    assert sum(len(v) for v in clusters.values()) == N
