#!/usr/bin/env python3

import argparse
import os

ID_SIZE = 8
DIM_SIZE = 4
BYTEORDER = 'little'


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('in_ids_file', type=str)
    parser.add_argument('in_emb_file', type=str)
    parser.add_argument('dim', type=int)
    parser.add_argument('out_ids_file', type=str)
    parser.add_argument('out_emb_file', type=str)
    return parser.parse_args()


def read_ids_file(id_file):
    ids = []
    with open(id_file, 'rb') as f:
        while True:
            next = f.read(ID_SIZE)
            if next == b'':
                break
            assert len(next) == ID_SIZE
            ids.append(int.from_bytes(next, BYTEORDER))
    return ids


def main(in_ids_file, in_emb_file, out_ids_file, out_emb_file, dim):
    assert dim > 0
    assert in_ids_file != out_ids_file
    assert in_emb_file != out_emb_file
    assert os.path.getsize(in_ids_file) / ID_SIZE == \
           os.path.getsize(in_emb_file) / (DIM_SIZE * dim)

    print('Reading old files:', in_ids_file, in_emb_file)
    ids = read_ids_file(in_ids_file)
    sorted_pos_and_id = sorted(enumerate(ids), key=lambda x: x[1])

    print('Writing new files:', out_ids_file, out_emb_file)
    emb_size = dim * DIM_SIZE
    with open(in_emb_file, 'rb') as ifs_emb, \
            open(out_ids_file, 'wb') as ofs_ids, \
            open(out_emb_file, 'wb') as ofs_emb:
        for orig_idx, emb_id in sorted_pos_and_id:
            # Write the ids file
            ofs_ids.write(emb_id.to_bytes(ID_SIZE, BYTEORDER))

            # Copy the embedding
            ifs_emb.seek(orig_idx * emb_size)
            emb = ifs_emb.read(emb_size)
            assert len(emb) == emb_size
            ofs_emb.write(emb)

    assert os.path.getsize(in_ids_file) == os.path.getsize(out_ids_file)
    assert os.path.getsize(in_emb_file) == os.path.getsize(out_emb_file)
    print('Done!')


if __name__ == '__main__':
    main(**vars(get_args()))
