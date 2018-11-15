#![feature(specialization)]

extern crate rayon;
extern crate pyo3;
extern crate memmap;
extern crate byteorder;

use rayon::prelude::*;
use pyo3::prelude::*;
use pyo3::exceptions;
use std::iter::Sum;
use std::mem;
use std::fs::File;
use std::io::{Cursor, Read};
use byteorder::{ReadBytesExt, LittleEndian};
use memmap::{MmapOptions, Mmap};

pub type Id = u64;

fn read_id_file(fname: String) -> Option<Vec<Id>> {
    let mut buf = Vec::new();
    match File::open(&fname) {
        Ok(mut f) => {
            f.read_to_end(&mut buf).expect("Failed to read id file");
            let mut ids = vec![0u64; buf.len() / mem::size_of::<u64>()];
            let mut rdr = Cursor::new(buf);
            rdr.read_u64_into::<LittleEndian>(&mut ids).unwrap();
            Some(ids)
        },
        Err(_) => None
    }
}

#[pyclass]
struct EmbeddingData {
    ids: Vec<Id>,
    data: Mmap,
    dim: usize
}

#[pymethods]
impl EmbeddingData {

    fn _read(&self, i: usize) -> Vec<f32> {
        let dim_size = std::mem::size_of::<f32>();
        let ofs = i * self.dim * dim_size;
        let mut rdr = Cursor::new(&self.data[ofs..ofs + self.dim * dim_size]);
        let mut ret = vec![0.0f32; self.dim];
        rdr.read_f32_into::<LittleEndian>(&mut ret).unwrap();
        ret
    }

    fn _dists(&self, xs: Vec<Vec<f32>>) -> Vec<(Id, f32)> {
        let mut dists: Vec<(Id, f32)> = self.ids.par_iter().enumerate().map(|(i, id)| {
            let z: Vec<f32> = self._read(i);
            (
                *id,
                xs.iter().map(
                    |x| f32::sum(x.iter().zip(z.iter()).map(
                        |(a, b)| (a - b).powi(2)
                    )).sqrt()
                ).fold(1./0., f32::min)
            )
        }).collect();
        dists.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap());
        dists
    }

    fn get(&self, id: Id) -> PyResult<Vec<f32>> {
        let idx = self.ids.binary_search(&id).unwrap();
        Ok(self._read(idx))
    }

    fn nn(&self, xs: Vec<Vec<f32>>, k: usize, threshold: f32) -> PyResult<Vec<(Id, f32)>> {
        if xs.len() == 0 {
            Err(exceptions::ValueError::py_err("No input"))
        } else {
            Ok(self._dists(xs).into_iter().filter(
                |(_, d)| *d <= threshold
            ).take(k).collect())
        }
    }

    fn nn_by_id(&self, ids: Vec<Id>, k: usize, threshold: f32) -> PyResult<Vec<(Id, f32)>> {
        if ids.len() == 0 {
            Err(exceptions::ValueError::py_err("No input"))
        } else {
            let xs: Vec<Vec<f32>> = ids.iter().map(
                |&id| self._read(self.ids.binary_search(&id).unwrap())
            ).collect();
            Ok(self._dists(xs).into_iter().filter(
                |(_, d)| *d <= threshold
            ).take(k).collect())
        }
    }

    fn count(&self) -> PyResult<usize> {
        Ok(self.ids.len())
    }

    #[new]
    unsafe fn __new__(obj: &PyRawObject, id_file: String, data_file: String, dim: usize) -> PyResult<()> {
        match read_id_file(id_file) {
            Some(ids) => {
                let mmap = MmapOptions::new().map(&File::open(&data_file)?);
                match mmap {
                    Ok(m) => obj.init(|_| EmbeddingData { ids: ids, data: m, dim: dim }),
                    Err(s) => Err(exceptions::Exception::py_err(s.to_string()))
                }
            },
            None => Err(exceptions::Exception::py_err("Failed to read ids"))
        }

    }
}

#[pymodinit]
fn rs_embed(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<EmbeddingData>()?;
    Ok(())
}
