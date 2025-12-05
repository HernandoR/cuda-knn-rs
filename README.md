# Rust-CUDA GPU KNN

GPU-accelerated batch point cloud KNN classifier implemented in Rust with Python bindings.

## Features

- **Dual-mode nearest neighbor search**:
  - Fixed K nearest neighbors: Returns top-K nearest neighbors for each query point
  - Range-KNN (range + count limited): Returns neighbors within distance threshold, limited to at most K neighbors

- **High performance**: Parallel processing for batch operations using Rayon

- **Batch processing**: Native support for batch processing of point clouds
  - Training data: coordinates (B×N×T), labels (B×N×1)
  - Query data: coordinates (B×M×T)
  - Output: (labels matrix, indices matrix), shape=(B×M×K)

- **Zero-copy data exchange**: Through Rust-Python bindings (PyO3+NumPy)

- **Easy to use**: Simple Python API (initialize → fit → predict/predict_range)

## Requirements

- Python 3.8+
- Rust 1.60+
- NumPy

## Installation

### From source (using maturin)

```bash
# Install maturin
pip install maturin

# Build and install
maturin develop --release
```

### For development

```bash
pip install maturin numpy pytest scikit-learn
maturin develop
```

## Usage

### Fixed K Nearest Neighbors

```python
import numpy as np
from rust_gpu_knn import GPUBatchPointCloudKNN

# Initialize KNN (k=default number of neighbors, device_id for future GPU support)
knn = GPUBatchPointCloudKNN(k=5, device_id=0)

# Prepare batch data (B=2 batches, N=200000 training points, T=3 dimensions)
x_train_coords = np.random.rand(2, 200000, 3).astype(np.float32)
y_train = np.random.randint(0, 5, (2, 200000, 1)).astype(np.int64)
x_test_coords = np.random.rand(2, 30000, 3).astype(np.float32)

# Fit model
knn.fit(x_train_coords, y_train)

# Predict fixed K nearest neighbors
topk_labels, topk_indices = knn.predict(x_test_coords, k=5)
print("Labels shape:", topk_labels.shape)  # (2, 30000, 5)
print("Indices shape:", topk_indices.shape)  # (2, 30000, 5)
```

### Range + Count Limited Nearest Neighbors (range-KNN)

```python
# Parameters for range-KNN
k = 8  # Maximum number of neighbors
distance_threshold = 0.1  # L2 squared distance threshold

# Predict range-KNN: returns neighbors within distance threshold,
# limited to at most k neighbors (closest ones first)
range_labels, range_indices = knn.predict_range(
    x_test_coords,
    k=k,
    distance_threshold=distance_threshold
)

print("Range-KNN labels shape:", range_labels.shape)  # (2, 30000, 8)
print("Range-KNN indices shape:", range_indices.shape)  # (2, 30000, 8)

# Filter invalid entries (-1 indicates no neighbor found)
valid_labels = range_labels[0, 0, :][range_labels[0, 0, :] != -1]
print("Valid neighbors for first query point:", valid_labels)
```

## API Reference

### `GPUBatchPointCloudKNN(k, device_id=0)`

Initialize the KNN classifier.

- `k` (int): Default number of nearest neighbors
- `device_id` (int): GPU device ID (for future CUDA support, default: 0)

### `fit(x_train_coords, y_train)`

Fit the model with training data.

- `x_train_coords` (ndarray[float32]): Training coordinates, shape (B, N, T)
- `y_train` (ndarray[int64]): Training labels, shape (B, N, 1)

### `predict(x_test_coords, k=None)`

Predict top-K nearest neighbors.

- `x_test_coords` (ndarray[float32]): Test coordinates, shape (B, M, T)
- `k` (int, optional): Number of neighbors (defaults to initialization value)

Returns:
- `labels` (ndarray[int64]): Neighbor labels, shape (B, M, K), -1 for invalid
- `indices` (ndarray[int64]): Neighbor indices, shape (B, M, K), -1 for invalid

### `predict_range(x_test_coords, k, distance_threshold)`

Predict neighbors within distance threshold.

- `x_test_coords` (ndarray[float32]): Test coordinates, shape (B, M, T)
- `k` (int): Maximum number of neighbors
- `distance_threshold` (float): L2 squared distance threshold

Returns:
- `labels` (ndarray[int64]): Neighbor labels, shape (B, M, K), -1 for invalid
- `indices` (ndarray[int64]): Neighbor indices, shape (B, M, K), -1 for invalid

## Testing

```bash
# Run Rust tests
cargo test

# Run Python tests (after installing with maturin develop)
pytest tests/
```

## Technical Details

### Distance Metric

The implementation uses L2 squared distance (Euclidean distance squared):

```
d(a, b) = Σ(a_i - b_i)²
```

Note: The `distance_threshold` parameter in `predict_range` uses squared distance.
If your data coordinates are in range [0, 1], typical threshold values are 0.01-0.1.

### Core Algorithms

1. **compute_distances**: Computes L2 squared distances between all query and training points
2. **select_topk**: Selects top-K nearest neighbors for each query point (fixed K mode)
3. **select_range_topk**: Filters neighbors within threshold, then selects top-K (range-KNN mode)

## License

MIT

