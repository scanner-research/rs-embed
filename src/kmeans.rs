#![feature(specialization)]
/* From here: https://github.com/genbattle/rkm */

use std::ops::Add;
use std::cmp::PartialOrd;
use std::fmt::Debug;
use std::marker::Sync;
use ndarray::{Array2, ArrayView1, ArrayView2, Ix, Axis, ScalarOperand};
use rand::Rng;
use rand::distributions::{Weighted, WeightedChoice, Distribution};
use num::{NumCast, Zero, Float};
use rayon::prelude::*;
use ndarray_parallel::prelude::*;
use rand::prelude::*;

/*
Numeric value trait, defines the types that can be used for the value of each dimension in a
data point.
*/
pub trait Value: ScalarOperand + Add + Zero + Float + NumCast + PartialOrd + Copy + Debug + Sync + Send {}
impl<T> Value for T where T: ScalarOperand + Add + Zero + Float + NumCast + PartialOrd + Copy + Debug + Sync + Send {}

/*
Find the distance between two data points, given as Array rows.
*/
fn distance_squared<V: Value>(point_a: &ArrayView1<V>, point_b: &ArrayView1<V>) -> V {
    let mut distance = V::zero();
    for i in 0..point_a.shape()[0] {
        let delta = point_a[i] - point_b[i];
        distance = distance + (delta * delta)
    }
    return distance;
}

/*
Find the shortest distance between each data point and any of a set of mean points.
*/
fn closest_distance<V: Value>(means: &ArrayView2<V>, data: &ArrayView2<V>) -> Vec<V> {
    data.outer_iter().into_par_iter().map(|d|{
        let mut iter = means.outer_iter();
        let mut closest = distance_squared(&d, &iter.next().unwrap());
        for m in iter {
            let distance = distance_squared(&d, &m);
            if distance < closest {
                closest = distance;
            }
        }
        closest
    }).collect()
}

/*
This is a mean initialization method based on the [kmeans++](https://en.wikipedia.org/wiki/K-means%2B%2B)
initialization algorithm.
*/
fn initialize_plusplus<V: Value>(data: &ArrayView2<V>, k: usize) -> Array2<V> {
    assert!(k > 1);
    assert!(data.dim().0 > 0);
    let mut means = Array2::zeros((k as usize, data.shape()[1]));
    let mut rng = SmallRng::from_rng(rand::thread_rng()).unwrap();
    let data_len = data.shape()[0];
    let chosen = rng.gen_range(0, data_len) as isize;
    means.slice_mut(s![0..1, ..]).assign(&data.slice(s![chosen..(chosen + 1), ..]));
    for i in 1..k as isize {
		// Calculate the distance to the closest mean for each data point
        let distances = closest_distance(&means.slice(s![0..i, ..]).view(), &data.view());
        // Pick a random point weighted by the distance from existing means
        let distance_sum: f64 = distances.iter().fold(0.0f64, |sum, d|{
            sum + num::cast::<V, f64>(*d).unwrap()
        });
        let mut weights: Vec<Weighted<usize>> = distances.par_iter().zip(0..data_len).map(|p|{
            Weighted{weight: ((num::cast::<V, f64>(*p.0).unwrap() / distance_sum) * ((std::u32::MAX) as f64)).floor() as u32, item: p.1}
        }).collect();
        let mut chooser = WeightedChoice::new(&mut weights);
        let chosen = chooser.sample(&mut rng) as isize;
        means.slice_mut(s![i..(i + 1), ..]).assign(&data.slice(s![chosen..(chosen + 1), ..]));
    }
    means
}

/*
Find the closest mean to a particular data point.
*/
fn closest_mean<V: Value>(point: &ArrayView1<V>, means: &ArrayView2<V>) -> Ix {
    assert!(means.dim().0 > 0);
    let mut iter = means.outer_iter().enumerate();
    if let Some(compare) = iter.next() {
        let mut index = compare.0;
        let mut shortest_distance = distance_squared(point, &compare.1);
        for compare in iter {
            let distance = distance_squared(point, &compare.1);
            if distance < shortest_distance {
                shortest_distance = distance;
                index = compare.0;
            }
        }
        return index;
    }
    return 0; // Should never hit this due to the assertion of the precondition
}

/*
Calculate the index of the mean each data point is closest to (euclidean distance).
*/
fn calculate_clusters<V: Value>(data: &ArrayView2<V>, means: &ArrayView2<V>) -> Vec<Ix> {
    data.outer_iter().into_par_iter()
    .map(|point|{
        closest_mean(&point.view(), means)
    })
    .collect()
}

/*
Calculate means based on data points and their cluster assignments.
*/
fn calculate_means<V: Value>(data: &ArrayView2<V>, clusters: &Vec<Ix>, old_means: &ArrayView2<V>, k: usize) -> Array2<V> {
    // TODO: replace old_means parameter with just its dimension, or eliminate it completely; that's all we need
    let (mut means, counts) = clusters.par_iter()
        .zip(data.outer_iter().into_par_iter())
        .fold(||(Array2::zeros(old_means.dim()), vec![0; k]), |mut totals, point|{
            {
                let mut sum = totals.0.subview_mut(Axis(0), *point.0);
                let new_sum = &sum + &point.1;
                sum.assign(&new_sum);
                // TODO: file a bug about how + and += work with ndarray
            }
            totals.1[*point.0] += 1;
            totals
        })
        .reduce(||(Array2::zeros(old_means.dim()), vec![0; k]), |new_means, subtotal|{
            let total = new_means.0 + subtotal.0;
            let count = new_means.1.iter().zip(subtotal.1.iter()).map(|counts|{
                counts.0 + counts.1
            }).collect();
            (total, count)
        });
    for i in 0..k {
        let mut sum = means.subview_mut(Axis(0), i);
        let new_mean = &sum / V::from(counts[i]).unwrap();
        sum.assign(&new_mean);
    }
    means
}

/*
Calculate means and cluster assignments for the given data and number of clusters (k).
Returns a tuple containing the means (as a 2D ndarray) and a `Vec` of indices that
map into the means ndarray and correspond elementwise to each input data point to give
the cluster assignments for each data point.
*/
pub fn kmeans_lloyd<V: Value>(data: &ArrayView2<V>, k: usize, max_iterations: usize) ->
    Result<(Array2<V>, Vec<usize>), &'static str> {
    assert!(k > 1);
    assert!(data.dim().0 >= k);

    let mut old_means = initialize_plusplus(data, k);
    let mut clusters = calculate_clusters(data, &old_means.view());
    let mut means = calculate_means(data, &clusters, &old_means.view(), k);

    let mut i = 0;
    while means != old_means {
        if i >= max_iterations {
            return Err("max iterations reached");
        }
        clusters = calculate_clusters(data, &means.view());
        old_means = means;
        means = calculate_means(data, &clusters, &old_means.view(), k);
        i += 1;
    }
    Ok((means, clusters))
}
