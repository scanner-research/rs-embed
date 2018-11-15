#!/usr/bin/env python3

import argparse
import os
import time
import traceback

from rs_embed import EmbeddingData

DEFAULT_K = 10


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('id_path', type=str, help='Binary file containing the ids')
    p.add_argument('emb_path', type=str,
                   help='Binary file containing the embeddings')
    p.add_argument('-k', dest='k', type=int, default=DEFAULT_K,
                   help='Number of nearest neighbors. Default: {}'.format(
                        DEFAULT_K))
    return p.parse_args()


def search(ids, k, emb_data):
    start_time = time.time()
    nn = emb_data.nn_by_id(ids, k, float('inf'))
    print([(a, round(b, 3)) for a, b in nn])
    print('Elapsed time: {:0.2f}s'.format(time.time() - start_time))


def main(id_path, emb_path, k):
    id_file_size = os.path.getsize(id_path)
    assert id_file_size % 8 == 0, \
        'Id file size is not a multiple of sizeof(u64)'
    n = int(id_file_size / 8)
    emb_file_size = os.path.getsize(emb_path)
    assert emb_file_size % 4 == 0, \
        'Embedding file size is a multiple of sizeof(f32)'
    d = int((emb_file_size / 4) / (id_file_size / 8))
    assert emb_file_size % d == 0, \
        'Embedding file size is a multiple of d={}'.format(d)

    print('Count:', n)
    print('Dimension:', d)

    emb_data = EmbeddingData(id_path, emb_path, d)
    assert emb_data.count() == n, \
        'Count does not match expected: {} != {}'.format(n, emb_data.count())

    print('Enter one or more ids (separated by ","s)')
    while True:
        line = input('> ').strip()
        if line == '':
            break
        try:
            ids = [int(i.strip()) for i in line.split(',')]
            search(ids, k, emb_data)
        except KeyboardInterrupt:
            break
        except Exception:
            traceback.print_exc()


if __name__ == '__main__':
    main(**vars(get_args()))
