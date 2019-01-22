import os
import random
import struct
import pytest
import numpy as np

from rs_embed import EmbeddingData


CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
ID_PATH = os.path.join(CURRENT_DIR, '.test_ids.bin')
DATA_PATH = os.path.join(CURRENT_DIR, '.test_data.bin')

DIM = 128
N = 10000


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


def test_exists():
    emb_data = EmbeddingData(ID_PATH, DATA_PATH, DIM)
    ids = [0, 1, 100, 9999, 10000, 10001, 100002]
    expected = [id < N for id in ids]
    assert emb_data.exists(ids) == expected


def test_sample():
    emb_data = EmbeddingData(ID_PATH, DATA_PATH, DIM)
    for k in [0, 1, 2, 5, 10, 1000]:
        samples = emb_data.sample(k)
        assert len(samples) == k
        assert all(emb_data.exists(samples))


def test_get():
    emb_data = EmbeddingData(ID_PATH, DATA_PATH, DIM)
    batch_result = emb_data.get(list(range(N)))
    assert len(batch_result) == N
    for i in range(N):
        j, v = emb_data.get([i])[0]
        assert j == i
        assert len(v) == DIM
        assert batch_result[i] == (i, v)


def test_mean():
    emb_data = EmbeddingData(ID_PATH, DATA_PATH, DIM)
    mean = emb_data.mean(list(range(N)))
    # Dummy data is zero-mean
    assert np.allclose(mean, [0] * len(mean), rtol=0.01, atol=0.01)


def test_dist():
    emb_data = EmbeddingData(ID_PATH, DATA_PATH, DIM)
    n1, n2 = 10, 100
    ids1 = list(range(n1))
    ids2 = list(range(n2))
    dists = emb_data.dist_by_id(ids1, ids2)
    assert len(dists) == len(ids2)
    assert np.allclose(dists[:n1], 0)

    embs1 = [v for _, v in emb_data.get(ids1)]
    dists2 = emb_data.dist(embs1, ids2)
    assert dists == dists2


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


def test_logreg():
    emb_data = EmbeddingData(ID_PATH, DATA_PATH, DIM)
    train_x = list(range(N))
    train_y = [float(i % 2) for i in range(N)]
    weights = emb_data.logreg(train_x, train_y, num_epochs=20,
                              learning_rate=0.1, l2_penalty=0.01,
                              l1_penalty=0.)

    pred1 = emb_data.logreg_predict(weights, min_thresh=-1, max_thresh=2)
    pred2 = emb_data.logreg_predict(weights, min_thresh=-1, max_thresh=2)
    for p1, p2 in zip(sorted(pred1), sorted(pred2)):
        i1, s1 = p1
        i2, s2 = p2
        assert i1 == i2 and np.isclose(s1, s2), \
            'Predictions from saved model do not match'


def test_knn():
    emb_data = EmbeddingData(ID_PATH, DATA_PATH, DIM)
    train_x = list(range(N))
    train_y = [float(i % 2) for i in range(N)]
    pred = emb_data.knn_predict(train_x, train_y, 5, min_thresh=-1,
                                max_thresh=2)
    assert len(pred) == N
    assert all(a >= 0. and a <= 1. for _, a in pred)
    # Make sure that the model does predict both classes
    assert sum(a > 0.5 for _, a in pred) > N / 4
    assert sum(a < 0.5 for _, a in pred) > N / 4
