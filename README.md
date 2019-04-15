# rs-embed

Python module for operations on vector embeddings.

## Install

First, make sure Rust is installed. Run `rustup override set nightly` inside the directory where the repository is cloned. Next, run `python3 setup.py install --user`.

## Tests

Run `pytest -v tests`.

## Data format

There are two files: an ids file and data file. The ids file is an array of u64 ids corresponding to the vectors and the data file is an array of f32 values with total length d times the length of the ids file.
