#![feature(specialization)]

extern crate rayon;
extern crate rand;
extern crate pyo3;
extern crate memmap;
extern crate byteorder;
extern crate rkm;
extern crate ndarray;
extern crate rustlearn;

use rayon::prelude::*;
use pyo3::prelude::*;
use pyo3::exceptions;
use rand::Rng;
use std::iter::Sum;
use std::mem;
use std::fs::File;
use std::io::{Cursor, Read};
use byteorder::{ReadBytesExt, LittleEndian};
use memmap::{MmapOptions, Mmap};
use rkm::kmeans_lloyd;
use rustlearn::prelude::*;
use rustlearn::linear_models::sgdclassifier::Hyperparameters;

pub type Id = u64;
pub type Embedding = Vec<f32>;

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

    fn _read(&self, i: usize) -> Embedding {
        let dim_size = std::mem::size_of::<f32>();
        let ofs = i * self.dim * dim_size;
        let mut rdr = Cursor::new(&self.data[ofs..ofs + self.dim * dim_size]);
        let mut ret = vec![0.0f32; self.dim];
        rdr.read_f32_into::<LittleEndian>(&mut ret).unwrap();
        ret
    }

    fn _dists(&self, xs: Vec<Embedding>, threshold: f32) -> Vec<(Id, f32)> {
        let mut dists: Vec<(Id, f32)> = self.ids.par_iter().enumerate().map(|(i, id)| {
            let z: Embedding = self._read(i);
            (
                *id,
                xs.iter().map(
                    |x| f32::sum(x.iter().zip(z.iter()).map(
                        |(a, b)| (a - b).powi(2)
                    )).sqrt()
                ).fold(1./0., f32::min)
            )
        }).filter(
            |(_, d)| *d <= threshold
        ).collect();
        dists.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap());
        dists
    }

    fn exists(&self, ids: Vec<Id>) -> PyResult<Vec<bool>> {
        Ok(ids.par_iter()
            .map(|id| self.ids.binary_search(&id).is_ok())
            .collect())
    }

    fn sample(&self, k: usize) -> PyResult<Vec<Id>> {
        let mut rng = rand::thread_rng();
        let mut ids: Vec<Id> = Vec::with_capacity(k);
        for _ in 0..k {
            ids.push(*(rng.choose(&self.ids).unwrap()));
        }
        Ok(ids)
    }

    fn get(&self, ids: Vec<Id>) -> PyResult<Vec<(Id, Embedding)>> {
        Ok(ids.par_iter()
            .map(|id| (id, self.ids.binary_search(&id)))
            .filter(|(_, r)| r.is_ok())
            .map(|(id, r)| (*id, self._read(r.unwrap())))
            .collect())
    }

    fn nn(&self, xs: Vec<Embedding>, k: usize, threshold: f32) -> PyResult<Vec<(Id, f32)>> {
        if xs.len() == 0 {
            Err(exceptions::ValueError::py_err("No input"))
        } else {
            Ok(self._dists(xs, threshold).into_iter().take(k).collect())
        }
    }

    fn nn_by_id(&self, ids: Vec<Id>, k: usize, threshold: f32) -> PyResult<Vec<(Id, f32)>> {
        if ids.len() == 0 {
            Err(exceptions::ValueError::py_err("No input"))
        } else {
            let xs: Vec<Embedding> = ids.iter().map(
                |&id| self._read(self.ids.binary_search(&id).unwrap())
            ).collect();
            Ok(self._dists(xs, threshold).into_iter().take(k).collect())
        }
    }

    fn count(&self) -> PyResult<usize> {
        Ok(self.ids.len())
    }

    fn kmeans(&self, ids: Vec<Id>, k: usize) -> PyResult<Vec<(Id, usize)>> {
        let ids_and_embs: Vec<(Id, Embedding)> = ids.par_iter()
            .map(|id| (id, self.ids.binary_search(&id)))
            .filter(|(_, r)| r.is_ok())
            .map(|(id, r)| (*id, self._read(r.unwrap())))
            .collect();
        let n = ids_and_embs.len();
        let mut data = ndarray::Array2::<f64>::zeros((n, self.dim));
        for i in 0..n {
            for j in 0..self.dim {
                data[[i, j]] = ids_and_embs[i].1[j] as f64;
            }
        }
        let (_, clusters) = kmeans_lloyd(&data.view(), k);
        Ok(clusters.iter().enumerate().map(|(id, c)| (ids_and_embs[id].0, *c)).collect())
    }

    fn logreg(&self, ids: Vec<Id>, labels: Vec<f32>, min_thresh: f32, max_thresh: f32)
              -> PyResult<Vec<(Id, f32)>> {
        if ids.len() != labels.len() {
            return Err(exceptions::ValueError::py_err("ids.len() != labels.len()"));
        }
        let embs: Vec<(Embedding, f32)> = ids.par_iter()
            .zip(labels.par_iter())
            .map(|(id, label)| (self.ids.binary_search(&id), *label))
            .filter(|(r, _)| r.is_ok())
            .map(|(r, label)| (self._read(r.unwrap()), label))
            .collect();
        if embs.len() == 0 {
            return Err(exceptions::ValueError::py_err("No training examples"));
        }

        let mut x = Array::zeros(embs.len(), self.dim);
        let mut y = Array::zeros(embs.len(), 1);
        for i in 0..embs.len() {
            y.set(i, 0, embs[i].1);
            for j in 0..self.dim {
                x.set(i, j, embs[i].0[j]);
            }
        }

        let mut model = Hyperparameters::new(self.dim)
            .learning_rate(0.1)
            .l2_penalty(0.1)
            .l1_penalty(0.0)
            .one_vs_rest();

        if !model.fit(&x, &y).is_ok() {
            return Err(exceptions::Exception::py_err("Failed to fit model"));
        }

        let mut predictions: Vec<(Id, f32)> = self.ids.par_iter().enumerate().map(|(i, id)| {
            let z: Embedding = self._read(i);
            let mut x = Array::zeros(1, self.dim);
            for i in 0..self.dim {
                x.set(0, i, z[i]);
            }
            (*id, model.decision_function(&x).expect("Failed to predict").get(0, 0) as f32)
        }).filter(|(_, s)| min_thresh <= *s && *s <= max_thresh).collect();
        predictions.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap());
        Ok(predictions)
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
