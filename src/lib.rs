#![feature(specialization)]

extern crate rayon;
extern crate rand;
extern crate pyo3;
extern crate memmap;
extern crate byteorder;
extern crate rkm;
extern crate ndarray;
extern crate rustlearn;
extern crate is_sorted;

use rayon::prelude::*;
use pyo3::prelude::*;
use pyo3::exceptions;
use rand::{Rng, thread_rng};
use std::iter::Sum;
use std::mem;
use std::fs::File;
use std::io::{Cursor, Read};
use byteorder::{ReadBytesExt, LittleEndian};
use memmap::{MmapOptions, Mmap};
use rkm::kmeans_lloyd;
use rustlearn::prelude::*;
use rustlearn::linear_models::sgdclassifier::Hyperparameters;
use is_sorted::IsSorted;

pub type Id = u64;
pub type Embedding = Vec<f32>;
pub type LogRegModel = Vec<Vec<f32>>;

fn read_id_file(fname: String) -> Option<Vec<Id>> {
    let mut buf = Vec::new();
    match File::open(&fname) {
        Ok(mut f) => {
            f.read_to_end(&mut buf).expect("Failed to read id file");
            let mut ids = vec![0u64; buf.len() / mem::size_of::<u64>()];
            let mut rdr = Cursor::new(buf);
            rdr.read_u64_into::<LittleEndian>(&mut ids).unwrap();
            assert!(ids.iter().is_sorted());
            Some(ids)
        },
        Err(_) => None
    }
}

fn l2_dist(v: &Embedding, w: &Embedding) -> f32 {
    f32::sum(
        v.iter().zip(w.iter()).map(
            |(a, b)| (a - b).powi(2)
        )
    ).sqrt()
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

struct EmbeddingDataImpl {
    ids: Vec<Id>,
    data: Mmap,
    dim: usize
}

impl EmbeddingDataImpl {
    // Internal helpers not exposed to Python and Pyo3

    fn read(&self, i: usize) -> Embedding {
        let dim_size = std::mem::size_of::<f32>();
        let ofs = i * self.dim * dim_size;
        let mut rdr = Cursor::new(&self.data[ofs..ofs + self.dim * dim_size]);
        let mut ret = vec![0.0f32; self.dim];
        rdr.read_f32_into::<LittleEndian>(&mut ret).unwrap();
        ret
    }

    fn get_ok(&self, ids: &Vec<Id>) -> Vec<(Id, Embedding)> {
        ids.par_iter()
            .map(|id| (id, self.ids.binary_search(&id)))
            .filter(|(_, r)| r.is_ok())
            .map(|(id, r)| (*id, self.read(r.unwrap())))
            .collect()
    }

    fn dists(&self, xs: &Vec<Embedding>, threshold: f32) -> Vec<(Id, f32)> {
        let mut dists: Vec<(Id, f32)> = self.ids.par_iter().enumerate().map(|(i, id)| {
            let z: Embedding = self.read(i);
            (
                *id,
                xs.iter().map(
                    |x| l2_dist(x, &z)
                ).fold(1./0., f32::min)
            )
        }).filter(
            |(_, d)| *d <= threshold
        ).collect();
        dists.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap());
        dists
    }
}

#[pyclass]
struct EmbeddingData {
    _internal: EmbeddingDataImpl
}

#[pymethods]
impl EmbeddingData {

    fn exists(&self, ids: Vec<Id>) -> PyResult<Vec<bool>> {
        Ok(ids.par_iter()
            .map(|id| self._internal.ids.binary_search(&id).is_ok())
            .collect())
    }

    fn sample(&self, k: usize) -> PyResult<Vec<Id>> {
        let mut rng = rand::thread_rng();
        let mut ids: Vec<Id> = Vec::with_capacity(k);
        for _ in 0..k {
            ids.push(*(rng.choose(&self._internal.ids).unwrap()));
        }
        Ok(ids)
    }

    fn get(&self, ids: Vec<Id>) -> PyResult<Vec<(Id, Embedding)>> {
        Ok(self._internal.get_ok(&ids))
    }

    fn mean(&self, ids: Vec<Id>) -> PyResult<Embedding> {
        let embs: Vec<(Id, Embedding)> = self._internal.get_ok(&ids);
        if embs.len() < ids.len() {
            Err(exceptions::ValueError::py_err("Not all ids were found"))
        } else {
            let n = embs.len();
            let mut mean: Embedding = vec![0.; self._internal.dim];
            for i in 0..n {
                for j in 0..self._internal.dim {
                    mean[j] = embs[i].1[j] / (n as f32);
                }
            }
            Ok(mean)
        }
    }

    fn dist(&self, xs: Vec<Embedding>, ids: Vec<Id>) -> PyResult<Vec<f32>> {
        if xs.len() == 0 {
            Err(exceptions::ValueError::py_err("No input"))
        } else {
            Ok(ids.par_iter().map(
                |&id| {
                    let v2opt = self._internal.ids.binary_search(&id);
                    match v2opt {
                        Ok(v2ofs) => {
                            let v2 = self._internal.read(v2ofs);
                            xs.iter().map(
                                |v1| l2_dist(v1, &v2)
                            ).fold(1./0., f32::min)
                        },
                        Err(_) => 1./0.
                    }
                }
            ).collect())
        }
    }

    fn dist_by_id(&self, ids1: Vec<Id>, ids2: Vec<Id>) -> PyResult<Vec<f32>> {
        if ids1.len() == 0 {
            Err(exceptions::ValueError::py_err("No input"))
        } else {
            let xs1: Vec<Embedding> = ids1.iter().map(
                |&id| self._internal.read(self._internal.ids.binary_search(&id).unwrap())
            ).collect();
            self.dist(xs1, ids2)
        }
    }

    fn nn(&self, xs: Vec<Embedding>, k: usize, threshold: f32) -> PyResult<Vec<(Id, f32)>> {
        if xs.len() == 0 {
            Err(exceptions::ValueError::py_err("No input"))
        } else {
            Ok(self._internal.dists(&xs, threshold).into_iter().take(k).collect())
        }
    }

    fn nn_by_id(&self, ids: Vec<Id>, k: usize, threshold: f32) -> PyResult<Vec<(Id, f32)>> {
        if ids.len() == 0 {
            Err(exceptions::ValueError::py_err("No input"))
        } else {
            let xs: Vec<Embedding> = ids.iter().map(
                |&id| self._internal.read(self._internal.ids.binary_search(&id).unwrap())
            ).collect();
            self.nn(xs, k, threshold)
        }
    }

    fn count(&self) -> PyResult<usize> {
        Ok(self._internal.ids.len())
    }

    fn kmeans(&self, ids: Vec<Id>, k: usize) -> PyResult<Vec<(Id, usize)>> {
        let ids_and_embs: Vec<(Id, Embedding)> = self._internal.get_ok(&ids);
        let n = ids_and_embs.len();
        let mut data = ndarray::Array2::<f64>::zeros((n, self._internal.dim));
        for i in 0..n {
            for j in 0..self._internal.dim {
                data[[i, j]] = ids_and_embs[i].1[j] as f64;
            }
        }
        let (_, clusters) = kmeans_lloyd(&data.view(), k);
        Ok(clusters.iter().enumerate().map(|(id, c)| (ids_and_embs[id].0, *c)).collect())
    }

    fn logreg(&self, ids: Vec<Id>, labels: Vec<f32>, min_thresh: f32, max_thresh: f32,
              num_epochs: usize, learning_rate: f32, l2_penalty: f32, l1_penalty: f32
    ) -> PyResult<(LogRegModel, Vec<(Id, f32)>)> {
        if ids.len() != labels.len() {
            return Err(exceptions::ValueError::py_err("ids.len() != labels.len()"));
        }
        let mut embs: Vec<(Embedding, f32)> = ids.par_iter()
            .zip(labels.par_iter())
            .map(|(id, label)| (self._internal.ids.binary_search(&id), *label))
            .filter(|(r, _)| r.is_ok())
            .map(|(r, label)| (self._internal.read(r.unwrap()), label))
            .collect();
        if embs.len() == 0 {
            return Err(exceptions::ValueError::py_err("No training examples"));
        }
        thread_rng().shuffle(&mut embs);

        // Embedding dimension + 2
        let feat_dim = self._internal.dim + 3;

        // Compute average embedding for each label
        let count_emb_0: usize = embs.iter().fold(0,
            |count, (_, label)| if *label < 0.5 { count } else { count + 1 });
        let count_emb_1: usize = embs.len() - count_emb_0;
        let mut avg_emb_0: Embedding = vec![0.; self._internal.dim];
        let mut avg_emb_1: Embedding = vec![0.; self._internal.dim];
        for i in 0..embs.len() {
            let l = embs[i].1;
            for j in 0..self._internal.dim {
                let v = embs[i].0[j];
                if l < 0.5 {
                    avg_emb_0[j] += v / (count_emb_0 as f32);
                } else {
                    avg_emb_1[j] += v / (count_emb_1 as f32);
                }
            }
        }

        // Instantiate training dataset
        let mut x = Array::zeros(embs.len(), feat_dim);
        let mut y = Array::zeros(embs.len(), 1);
        for i in 0..embs.len() {
            let l = embs[i].1;
            if l.is_nan() {
                return Err(exceptions::ValueError::py_err("Found NaN in the training labels"));
            }
            y.set(i, 0, l);
            for j in 0..self._internal.dim {
                let v = embs[i].0[j];
                if v.is_nan() {
                    return Err(exceptions::ValueError::py_err("Found NaN in the training data"));
                }
                x.set(i, j, v);
            }

            // Additional features
            x.set(i, self._internal.dim, l2_dist(&avg_emb_0, &embs[i].0));
            x.set(i, self._internal.dim + 1, l2_dist(&avg_emb_1, &embs[i].0));
            x.set(i, self._internal.dim + 2, 1.); // Bias term
        }

        // Instantiate model
        let mut model = Hyperparameters::new(feat_dim)
            .learning_rate(learning_rate)
            .l2_penalty(l2_penalty)
            .l1_penalty(l1_penalty)
            .build();

        for _ in 0..num_epochs {
            if !model.fit(&x, &y).is_ok() {
                return Err(exceptions::Exception::py_err("Failed to fit model"));
            }
        }

        // Read model weights
        let weights: Vec<f32> = model.get_coefficients().data().to_vec();
        if weights.iter().any(|v| v.is_nan()) {
            return Err(exceptions::ValueError::py_err(
                format!("Found NaN in weights: {:?}", weights)));
        }

        // Predict on all embeddings
        let mut predictions: Vec<(Id, f32)> = self._internal.ids.par_iter().enumerate().map(
            |(i, id)| {
                let z: Embedding = self._internal.read(i);
                let mut x = Array::zeros(1, feat_dim);
                for i in 0..self._internal.dim {
                    x.set(0, i, z[i]);
                }
                // Additional features
                x.set(0, self._internal.dim, l2_dist(&avg_emb_0, &z));
                x.set(0, self._internal.dim + 1, l2_dist(&avg_emb_1, &z));
                x.set(0, self._internal.dim + 2, 1.);
                (*id, model.decision_function(&x).expect("Failed to predict").get(0, 0) as f32)
            }
        ).filter(|(_, s)| !s.is_nan() && min_thresh <= *s && *s <= max_thresh).collect();

        if predictions.len() == 0 {
            return Err(exceptions::ValueError::py_err("Model failed to predict"));
        }
        predictions.par_sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap());
        Ok((vec![weights, avg_emb_0, avg_emb_1], predictions))
    }

    fn logreg_predict(&self, model: LogRegModel) -> PyResult<Vec<(Id, f32)>> {
        if model.len() != 3 {
            return Err(exceptions::ValueError::py_err("Invalid model"));
        }
        let weights: &Vec<f32> = &model[0];
        let avg_emb_0: &Embedding = &model[1];
        let avg_emb_1: &Embedding = &model[2];
        if avg_emb_0.len() != self._internal.dim || avg_emb_1.len() != self._internal.dim {
            return Err(exceptions::ValueError::py_err("Invalid model: bad centriods"));
        }
        if weights.len() != self._internal.dim + 3 {
            return Err(exceptions::ValueError::py_err("Invalid model: bad weights"));
        }

        // Predict on all embeddings
        let mut predictions: Vec<(Id, f32)> = self._internal.ids.par_iter().enumerate().map(
            |(i, id)| {
                let z: Embedding = self._internal.read(i);
                let mut score = 0.;
                for i in 0..self._internal.dim {
                    score += weights[i] * z[i];
                }
                // Additional features
                score += weights[self._internal.dim] * l2_dist(avg_emb_0, &z);
                score += weights[self._internal.dim + 1] * l2_dist(avg_emb_1, &z);
                // Bias term
                score += weights[self._internal.dim + 2];
                (*id, sigmoid(score))
            }
        ).collect();
        predictions.par_sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap());
        Ok(predictions)
    }

    #[new]
    unsafe fn __new__(obj: &PyRawObject, id_file: String, data_file: String, dim: usize) -> PyResult<()> {
        match read_id_file(id_file) {
            Some(ids) => {
                let mmap = MmapOptions::new().map(&File::open(&data_file)?);
                match mmap {
                    Ok(m) => obj.init(|_| EmbeddingData {
                        _internal: EmbeddingDataImpl { ids: ids, data: m, dim: dim }
                    }),
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
