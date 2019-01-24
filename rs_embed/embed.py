import os
from typing import List, Tuple

from .rs_embed import RsEmbeddingData


Embedding = List[float]
Id = int

ID_BYTES = 8
DIM_BYTES = 4


class EmbeddingData(object):

    def __init__(self, id_file: str, data_file: str, dim: int):
        EmbeddingData.check(id_file, data_file, dim)
        self._data_file = data_file
        self._id_file = id_file
        self._dim = dim
        self._data = RsEmbeddingData(id_file, data_file, dim)

    @property
    def data_file():
        return self._data_file

    @property
    def id_file():
        return self._id_file

    @property
    def d():
        return self._dim

    def count(self) -> int:
        """Number of embeddings"""
        return self._data.count()

    def ids(self, start: int, n: int) -> List[Id]:
        """Ids at index in the id file"""
        return self._data.ids(start, n)

    def exists(self, ids: List[Id]) -> List[bool]:
        """Returns whether ids exist"""
        return self._data.exists(ids)

    def sample(self, k: int) -> List[Id]:
        """Uniformly sample ids"""
        return self._data.sample(k)

    def get(self, ids: List[Id]) -> List[Tuple[int, Embedding]]:
        """Get embeddings for ids"""
        return self._data.get(ids)

    def mean(self, ids: List[Id]) -> Embedding:
        """Compute mean of embeddings"""
        return self._data.mean(ids)

    def dist(self, xs: List[Embedding], ids: List[Id]) -> Embedding:
        """Compute distance of ids to xs"""
        return self._data.dist(xs, ids)

    def dist_by_id(self, ids1: List[Id], ids2: List[Id]) -> Embedding:
        """Compute distance of ids2 to ids1"""
        return self._data.dist_by_id(ids1, ids2)

    def nn(
        self, xs: List[Embedding], k: int, threshold: float, sample=1
    ) -> List[Tuple[Id, float]]:
        """Compute distances to xs"""
        return self._data.nn(xs, k, threshold, sample)

    def nn_by_id(
        self, ids: List[Id], k: int, threshold: float, sample=1
    ) -> List[Tuple[Id, float]]:
        """Compute distances to ids"""
        return self._data.nn_by_id(ids, k, threshold, sample)

    def kmeans(self, ids: List[Id], k: int) -> List[Tuple[Id, int]]:
        """
        Compute kmeans clusters. Returns ids and cluster assignments.
        """
        return self._data.kmeans(ids, k)

    def logreg(
        self, ids: List[Id], labels: List[int],
        num_epochs=10, learning_rate=1., l2_penalty=0., l1_penalty=0.
    ) -> List[List[float]]:
        """Train binary logistic regressor"""
        assert max(labels) <= 1
        assert min(labels) >= 0
        return self._data.logreg(
            ids, labels, num_epochs, learning_rate, l2_penalty,
            l1_penalty)

    def logreg_predict(
        self, weights, min_thresh=0., max_thresh=1., sample=1,
        ids=None
    ) -> List[Tuple[Id, float]]:
        """
        Make binary predictions for using a logistic regressor.

        If ids=None, then predictions will be made on all embeddings.
        """
        return self._data.logreg_predict(
            weights, min_thresh, max_thresh, sample,
            [] if ids is None else ids)

    def knn_predict(
        self, train_ids: List[Id], labels: List[int], k: int,
        min_thresh=0., max_thresh=1., sample=1, ids=None
    ) -> List[Tuple[Id, float]]:
        """
        Make binary predictions using k-NN.

        If ids=None, then predictions will be made on all embeddings.
        """
        assert max(labels) <= 1
        assert min(labels) >= 0
        return self._data.knn_predict(
            train_ids, labels, k, min_thresh, max_thresh, sample,
            [] if ids is None else ids)

    @staticmethod
    def check(id_file: str, data_file: str, dim: int):
        """Sanity check the id and data files"""
        id_file_size = os.path.getsize(id_file)
        if id_file_size % ID_BYTES != 0:
            raise Exception('Id file size is not a multiple of sizeof(u64)')
        data_file_size = os.path.getsize(data_file)
        if data_file_size % DIM_BYTES != 0:
            raise Exception(
                'Data file size is not a multiple of sizeof(f32)')
        d = int((data_file_size / DIM_BYTES) / (id_file_size / ID_BYTES))
        if d != dim:
            raise Exception(
                'Inconsistent dimension size: d={}. Expected: {}'.format(
                 d, dim))
        if data_file_size % dim != 0:
            raise Exception(
                'Data file size is not a multiple of d={}'.format(dim))
