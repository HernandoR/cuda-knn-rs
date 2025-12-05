//! GPU-accelerated batch point cloud KNN classifier
//!
//! This module provides a high-performance KNN implementation that supports:
//! - Fixed K nearest neighbors
//! - Range + count limited nearest neighbors (range-KNN)
//!
//! The implementation uses parallel processing for batch operations.

#![allow(clippy::type_complexity)]
#![allow(clippy::useless_conversion)]

use ndarray::{Array2, Array3, ArrayView1, ArrayView3, Axis};
use numpy::{PyArray3, PyReadonlyArray3, ToPyArray};
use ordered_float::OrderedFloat;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::cmp::Ordering;

/// Type alias for KNN prediction results
type KnnPredictResult<'py> = PyResult<(Bound<'py, PyArray3<i64>>, Bound<'py, PyArray3<i64>>)>;

/// Internal representation of a neighbor with distance and index
#[derive(Clone, Copy)]
struct Neighbor {
    distance: f32,
    index: i64,
    label: i64,
}

impl Neighbor {
    fn new(distance: f32, index: i64, label: i64) -> Self {
        Self {
            distance,
            index,
            label,
        }
    }

    fn invalid() -> Self {
        Self {
            distance: f32::MAX,
            index: -1,
            label: -1,
        }
    }
}

impl PartialEq for Neighbor {
    fn eq(&self, other: &Self) -> bool {
        OrderedFloat(self.distance) == OrderedFloat(other.distance)
    }
}

impl Eq for Neighbor {}

impl PartialOrd for Neighbor {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Neighbor {
    fn cmp(&self, other: &Self) -> Ordering {
        OrderedFloat(self.distance).cmp(&OrderedFloat(other.distance))
    }
}

/// Compute squared L2 distance between two ndarray 1D views
#[inline]
fn l2_squared_distance_views(query: ArrayView1<f32>, train: ArrayView1<f32>) -> f32 {
    query
        .iter()
        .zip(train.iter())
        .map(|(ai, bi)| (ai - bi).powi(2))
        .sum()
}

/// Find top-k nearest neighbors for a single query point
fn find_topk_neighbors(
    query: ArrayView1<f32>,
    train_coords: &ArrayView3<f32>,
    train_labels: &Array2<i64>,
    batch_idx: usize,
    k: usize,
) -> Vec<Neighbor> {
    let n_train = train_coords.shape()[1];
    let mut neighbors: Vec<Neighbor> = Vec::with_capacity(n_train);

    for j in 0..n_train {
        let train_point = train_coords.slice(ndarray::s![batch_idx, j, ..]);
        let dist = l2_squared_distance_views(query, train_point);
        let label = train_labels[[batch_idx, j]];
        neighbors.push(Neighbor::new(dist, j as i64, label));
    }

    // Sort by distance and take top-k
    neighbors.sort();
    neighbors.truncate(k);

    // Pad with invalid neighbors if needed
    while neighbors.len() < k {
        neighbors.push(Neighbor::invalid());
    }

    neighbors
}

/// Find neighbors within distance threshold, limited to at most k neighbors
fn find_range_topk_neighbors(
    query: ArrayView1<f32>,
    train_coords: &ArrayView3<f32>,
    train_labels: &Array2<i64>,
    batch_idx: usize,
    k: usize,
    distance_threshold: f32,
) -> Vec<Neighbor> {
    let n_train = train_coords.shape()[1];
    let mut candidates: Vec<Neighbor> = Vec::new();

    // First pass: collect all neighbors within threshold
    for j in 0..n_train {
        let train_point = train_coords.slice(ndarray::s![batch_idx, j, ..]);
        let dist = l2_squared_distance_views(query, train_point);

        if dist <= distance_threshold {
            let label = train_labels[[batch_idx, j]];
            candidates.push(Neighbor::new(dist, j as i64, label));
        }
    }

    // Sort by distance and take top-k
    candidates.sort();
    candidates.truncate(k);

    // Pad with invalid neighbors if needed
    let mut result = candidates;
    while result.len() < k {
        result.push(Neighbor::invalid());
    }

    result
}

/// GPU Batch Point Cloud KNN Classifier
///
/// This class implements a KNN classifier optimized for batch processing of point clouds.
/// It supports both fixed-K nearest neighbor search and range-limited nearest neighbor search.
#[pyclass]
pub struct GPUBatchPointCloudKNN {
    /// Default number of neighbors
    k: usize,
    /// GPU device ID (for future CUDA support)
    #[allow(dead_code)]
    device_id: usize,
    /// Training coordinates: shape (B, N, T) where B=batch, N=points, T=dimensions
    train_coords: Option<Array3<f32>>,
    /// Training labels: shape (B, N)
    train_labels: Option<Array2<i64>>,
}

#[pymethods]
impl GPUBatchPointCloudKNN {
    /// Create a new GPUBatchPointCloudKNN instance
    ///
    /// # Arguments
    /// * `k` - Default number of nearest neighbors
    /// * `device_id` - GPU device ID (default: 0)
    #[new]
    #[pyo3(signature = (k, device_id = 0))]
    pub fn new(k: usize, device_id: usize) -> PyResult<Self> {
        if k == 0 {
            return Err(PyValueError::new_err("k must be greater than 0"));
        }
        Ok(Self {
            k,
            device_id,
            train_coords: None,
            train_labels: None,
        })
    }

    /// Fit the KNN model with training data
    ///
    /// # Arguments
    /// * `x_train_coords` - Training coordinates, shape (B, N, T), float32
    /// * `y_train` - Training labels, shape (B, N) or (B, N, 1), int64
    pub fn fit(
        &mut self,
        x_train_coords: PyReadonlyArray3<f32>,
        y_train: PyReadonlyArray3<i64>,
    ) -> PyResult<()> {
        let coords = x_train_coords.as_array().to_owned();
        let labels_3d = y_train.as_array();

        // Convert labels from (B, N, 1) to (B, N)
        let labels = if labels_3d.shape()[2] == 1 {
            labels_3d.index_axis(Axis(2), 0).to_owned()
        } else {
            return Err(PyValueError::new_err("y_train must have shape (B, N, 1)"));
        };

        // Validate shapes
        if coords.shape()[0] != labels.shape()[0] {
            return Err(PyValueError::new_err(
                "Batch size of coordinates and labels must match",
            ));
        }
        if coords.shape()[1] != labels.shape()[1] {
            return Err(PyValueError::new_err(
                "Number of training points must match between coordinates and labels",
            ));
        }

        self.train_coords = Some(coords);
        self.train_labels = Some(labels);

        Ok(())
    }

    /// Predict nearest neighbors for test points (fixed K mode)
    ///
    /// # Arguments
    /// * `x_test_coords` - Test coordinates, shape (B, M, T), float32
    /// * `k` - Number of neighbors (optional, defaults to initialization value)
    ///
    /// # Returns
    /// Tuple of (labels, indices), each with shape (B, M, K)
    #[pyo3(signature = (x_test_coords, k = None))]
    pub fn predict<'py>(
        &self,
        py: Python<'py>,
        x_test_coords: PyReadonlyArray3<f32>,
        k: Option<usize>,
    ) -> KnnPredictResult<'py> {
        let train_coords = self
            .train_coords
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Model not fitted. Call fit() first."))?;
        let train_labels = self
            .train_labels
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Model not fitted. Call fit() first."))?;

        let test_coords = x_test_coords.as_array();
        let k = k.unwrap_or(self.k);

        if k == 0 {
            return Err(PyValueError::new_err("k must be greater than 0"));
        }

        // Validate shapes
        let batch_size = test_coords.shape()[0];
        let n_test = test_coords.shape()[1];
        let n_dims = test_coords.shape()[2];

        if batch_size != train_coords.shape()[0] {
            return Err(PyValueError::new_err(
                "Batch size of test and train data must match",
            ));
        }
        if n_dims != train_coords.shape()[2] {
            return Err(PyValueError::new_err(
                "Coordinate dimensions must match between test and train data",
            ));
        }

        // Allocate output arrays
        let mut labels_out = Array3::<i64>::zeros((batch_size, n_test, k));
        let mut indices_out = Array3::<i64>::zeros((batch_size, n_test, k));

        // Process each batch in parallel
        let train_coords_view = train_coords.view();
        let results: Vec<Vec<Vec<Neighbor>>> = (0..batch_size)
            .into_par_iter()
            .map(|b| {
                (0..n_test)
                    .map(|m| {
                        let query = test_coords.slice(ndarray::s![b, m, ..]);
                        find_topk_neighbors(query, &train_coords_view, train_labels, b, k)
                    })
                    .collect()
            })
            .collect();

        // Copy results to output arrays
        for (b, batch_results) in results.iter().enumerate() {
            for (m, neighbors) in batch_results.iter().enumerate() {
                for (ki, neighbor) in neighbors.iter().enumerate() {
                    labels_out[[b, m, ki]] = neighbor.label;
                    indices_out[[b, m, ki]] = neighbor.index;
                }
            }
        }

        Ok((
            labels_out.to_pyarray_bound(py),
            indices_out.to_pyarray_bound(py),
        ))
    }

    /// Predict neighbors within distance threshold (range-KNN mode)
    ///
    /// Returns neighbors that are within `distance_threshold` (L2 squared distance),
    /// limited to at most `k` neighbors (taking the closest ones).
    ///
    /// # Arguments
    /// * `x_test_coords` - Test coordinates, shape (B, M, T), float32
    /// * `k` - Maximum number of neighbors
    /// * `distance_threshold` - Distance threshold (L2 squared distance)
    ///
    /// # Returns
    /// Tuple of (labels, indices), each with shape (B, M, K)
    /// Positions without valid neighbors are filled with -1
    pub fn predict_range<'py>(
        &self,
        py: Python<'py>,
        x_test_coords: PyReadonlyArray3<f32>,
        k: usize,
        distance_threshold: f32,
    ) -> KnnPredictResult<'py> {
        let train_coords = self
            .train_coords
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Model not fitted. Call fit() first."))?;
        let train_labels = self
            .train_labels
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Model not fitted. Call fit() first."))?;

        let test_coords = x_test_coords.as_array();

        if k == 0 {
            return Err(PyValueError::new_err("k must be greater than 0"));
        }
        if distance_threshold < 0.0 {
            return Err(PyValueError::new_err(
                "distance_threshold must be non-negative",
            ));
        }

        // Validate shapes
        let batch_size = test_coords.shape()[0];
        let n_test = test_coords.shape()[1];
        let n_dims = test_coords.shape()[2];

        if batch_size != train_coords.shape()[0] {
            return Err(PyValueError::new_err(
                "Batch size of test and train data must match",
            ));
        }
        if n_dims != train_coords.shape()[2] {
            return Err(PyValueError::new_err(
                "Coordinate dimensions must match between test and train data",
            ));
        }

        // Allocate output arrays with -1 (invalid) default
        let mut labels_out = Array3::<i64>::from_elem((batch_size, n_test, k), -1);
        let mut indices_out = Array3::<i64>::from_elem((batch_size, n_test, k), -1);

        // Process each batch in parallel
        let train_coords_view = train_coords.view();
        let results: Vec<Vec<Vec<Neighbor>>> = (0..batch_size)
            .into_par_iter()
            .map(|b| {
                (0..n_test)
                    .map(|m| {
                        let query = test_coords.slice(ndarray::s![b, m, ..]);
                        find_range_topk_neighbors(
                            query,
                            &train_coords_view,
                            train_labels,
                            b,
                            k,
                            distance_threshold,
                        )
                    })
                    .collect()
            })
            .collect();

        // Copy results to output arrays
        for (b, batch_results) in results.iter().enumerate() {
            for (m, neighbors) in batch_results.iter().enumerate() {
                for (ki, neighbor) in neighbors.iter().enumerate() {
                    labels_out[[b, m, ki]] = neighbor.label;
                    indices_out[[b, m, ki]] = neighbor.index;
                }
            }
        }

        Ok((
            labels_out.to_pyarray_bound(py),
            indices_out.to_pyarray_bound(py),
        ))
    }
}

/// Python module definition
#[pymodule]
fn rust_gpu_knn(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<GPUBatchPointCloudKNN>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array3};

    #[test]
    fn test_l2_squared_distance() {
        let a = Array1::from_vec(vec![0.0, 0.0, 0.0]);
        let b = Array1::from_vec(vec![1.0, 0.0, 0.0]);
        assert!((l2_squared_distance_views(a.view(), b.view()) - 1.0).abs() < 1e-6);

        let c = Array1::from_vec(vec![1.0, 1.0, 1.0]);
        assert!((l2_squared_distance_views(a.view(), c.view()) - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_neighbor_ordering() {
        let n1 = Neighbor::new(1.0, 0, 0);
        let n2 = Neighbor::new(2.0, 1, 1);
        assert!(n1 < n2);
    }

    #[test]
    fn test_find_topk_neighbors() {
        // Create simple test data: 1 batch, 5 training points, 2D
        let train_coords = Array3::<f32>::from_shape_vec(
            (1, 5, 2),
            vec![
                0.0, 0.0, // point 0
                1.0, 0.0, // point 1
                0.0, 1.0, // point 2
                1.0, 1.0, // point 3
                2.0, 2.0, // point 4
            ],
        )
        .unwrap();

        let train_labels = Array2::<i64>::from_shape_vec((1, 5), vec![0, 1, 2, 3, 4]).unwrap();

        let query = Array1::from_vec(vec![0.1, 0.1]); // Close to point 0
        let neighbors =
            find_topk_neighbors(query.view(), &train_coords.view(), &train_labels, 0, 3);

        assert_eq!(neighbors.len(), 3);
        assert_eq!(neighbors[0].index, 0); // Closest should be point 0
    }

    #[test]
    fn test_find_range_topk_neighbors() {
        let train_coords = Array3::<f32>::from_shape_vec(
            (1, 5, 2),
            vec![
                0.0, 0.0, // point 0 - distance 0.02 from query
                1.0, 0.0, // point 1 - distance ~0.82
                0.0, 1.0, // point 2 - distance ~0.82
                1.0, 1.0, // point 3 - distance ~1.62
                2.0, 2.0, // point 4 - distance ~7.22
            ],
        )
        .unwrap();

        let train_labels = Array2::<i64>::from_shape_vec((1, 5), vec![0, 1, 2, 3, 4]).unwrap();

        let query = Array1::from_vec(vec![0.1, 0.1]);
        // Threshold of 1.0 should include point 0 (dist ~0.02) but exclude others
        let neighbors = find_range_topk_neighbors(
            query.view(),
            &train_coords.view(),
            &train_labels,
            0,
            3,
            0.1, // Only point 0 should be within this threshold
        );

        assert_eq!(neighbors.len(), 3);
        assert_eq!(neighbors[0].index, 0); // Point 0 should be included
        assert_eq!(neighbors[1].index, -1); // Rest should be invalid
        assert_eq!(neighbors[2].index, -1);
    }
}
