"""Tests for the GPU KNN implementation."""

import numpy as np
import pytest


def test_basic_knn():
    """Test basic KNN functionality."""
    from rust_gpu_knn import GPUBatchPointCloudKNN

    # Create simple test data
    batch_size = 1
    n_train = 10
    n_test = 2
    n_dims = 3
    k = 3

    # Training data: random points
    np.random.seed(42)
    x_train = np.random.rand(batch_size, n_train, n_dims).astype(np.float32)
    y_train = np.arange(n_train).reshape(batch_size, n_train, 1).astype(np.int64)

    # Test data
    x_test = np.random.rand(batch_size, n_test, n_dims).astype(np.float32)

    # Create and fit KNN
    knn = GPUBatchPointCloudKNN(k=k, device_id=0)
    knn.fit(x_train, y_train)

    # Predict
    labels, indices = knn.predict(x_test)

    # Check output shapes
    assert labels.shape == (batch_size, n_test, k)
    assert indices.shape == (batch_size, n_test, k)

    # Check that indices are valid (0 to n_train-1 or -1)
    assert np.all((indices >= 0) & (indices < n_train) | (indices == -1))

    # Check that labels match indices
    for b in range(batch_size):
        for m in range(n_test):
            for ki in range(k):
                idx = indices[b, m, ki]
                if idx >= 0:
                    assert labels[b, m, ki] == y_train[b, idx, 0]


def test_knn_correct_ordering():
    """Test that neighbors are returned in correct distance order."""
    from rust_gpu_knn import GPUBatchPointCloudKNN

    # Create structured test data
    # Training points at known distances from origin
    x_train = np.array(
        [
            [
                [1.0, 0.0, 0.0],  # distance 1
                [2.0, 0.0, 0.0],  # distance 4
                [0.5, 0.0, 0.0],  # distance 0.25
                [3.0, 0.0, 0.0],  # distance 9
                [0.1, 0.0, 0.0],  # distance 0.01
            ]
        ],
        dtype=np.float32,
    )
    y_train = np.array([[[0], [1], [2], [3], [4]]], dtype=np.int64)

    # Query at origin - should find closest points first
    x_test = np.array([[[0.0, 0.0, 0.0]]], dtype=np.float32)

    knn = GPUBatchPointCloudKNN(k=5, device_id=0)
    knn.fit(x_train, y_train)

    labels, indices = knn.predict(x_test)

    # Expected order by distance: point 4 (0.01), point 2 (0.25),
    # point 0 (1), point 1 (4), point 3 (9)
    expected_indices = [4, 2, 0, 1, 3]
    np.testing.assert_array_equal(indices[0, 0], expected_indices)


def test_range_knn():
    """Test range-KNN functionality."""
    from rust_gpu_knn import GPUBatchPointCloudKNN

    # Training points at known distances from origin
    x_train = np.array(
        [
            [
                [0.1, 0.0, 0.0],  # squared distance 0.01
                [0.2, 0.0, 0.0],  # squared distance 0.04
                [0.5, 0.0, 0.0],  # squared distance 0.25
                [1.0, 0.0, 0.0],  # squared distance 1.0
                [2.0, 0.0, 0.0],  # squared distance 4.0
            ]
        ],
        dtype=np.float32,
    )
    y_train = np.array([[[0], [1], [2], [3], [4]]], dtype=np.int64)

    # Query at origin
    x_test = np.array([[[0.0, 0.0, 0.0]]], dtype=np.float32)

    knn = GPUBatchPointCloudKNN(k=5, device_id=0)
    knn.fit(x_train, y_train)

    # With threshold 0.1, should only include points with squared distance <= 0.1
    # This includes point 0 (0.01) and point 1 (0.04)
    labels, indices = knn.predict_range(x_test, k=5, distance_threshold=0.1)

    assert labels.shape == (1, 1, 5)
    assert indices.shape == (1, 1, 5)

    # First two should be valid, rest should be -1
    assert indices[0, 0, 0] in [0, 1]  # closest point
    assert indices[0, 0, 1] in [0, 1]  # second closest
    assert indices[0, 0, 2] == -1  # no more within threshold
    assert indices[0, 0, 3] == -1
    assert indices[0, 0, 4] == -1


def test_batch_processing():
    """Test batch processing with multiple batches."""
    from rust_gpu_knn import GPUBatchPointCloudKNN

    batch_size = 3
    n_train = 100
    n_test = 20
    n_dims = 3
    k = 5

    np.random.seed(42)
    x_train = np.random.rand(batch_size, n_train, n_dims).astype(np.float32)
    y_train = np.random.randint(0, 10, (batch_size, n_train, 1)).astype(np.int64)
    x_test = np.random.rand(batch_size, n_test, n_dims).astype(np.float32)

    knn = GPUBatchPointCloudKNN(k=k, device_id=0)
    knn.fit(x_train, y_train)

    labels, indices = knn.predict(x_test)

    assert labels.shape == (batch_size, n_test, k)
    assert indices.shape == (batch_size, n_test, k)


def test_dynamic_k():
    """Test that k can be changed at prediction time."""
    from rust_gpu_knn import GPUBatchPointCloudKNN

    x_train = np.random.rand(1, 50, 3).astype(np.float32)
    y_train = np.arange(50).reshape(1, 50, 1).astype(np.int64)
    x_test = np.random.rand(1, 5, 3).astype(np.float32)

    # Initialize with k=3
    knn = GPUBatchPointCloudKNN(k=3, device_id=0)
    knn.fit(x_train, y_train)

    # Predict with default k
    labels3, _ = knn.predict(x_test)
    assert labels3.shape == (1, 5, 3)

    # Predict with different k
    labels10, _ = knn.predict(x_test, k=10)
    assert labels10.shape == (1, 5, 10)


def test_invalid_inputs():
    """Test error handling for invalid inputs."""
    from rust_gpu_knn import GPUBatchPointCloudKNN

    # k must be > 0
    with pytest.raises(ValueError):
        GPUBatchPointCloudKNN(k=0)

    # Model must be fitted before prediction
    knn = GPUBatchPointCloudKNN(k=5)
    x_test = np.random.rand(1, 10, 3).astype(np.float32)
    with pytest.raises(ValueError):
        knn.predict(x_test)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
