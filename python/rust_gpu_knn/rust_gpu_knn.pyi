"""Type stubs for rust_gpu_knn module."""

from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

class GPUBatchPointCloudKNN:
    """GPU Batch Point Cloud KNN Classifier.

    This class implements a KNN classifier optimized for batch processing
    of point clouds. It supports both fixed-K nearest neighbor search and
    range-limited nearest neighbor search.

    Args:
        k: Default number of nearest neighbors.
        device_id: GPU device ID (default: 0).

    Example:
        >>> import numpy as np
        >>> from rust_gpu_knn import GPUBatchPointCloudKNN
        >>> knn = GPUBatchPointCloudKNN(k=5, device_id=0)
        >>> x_train = np.random.rand(2, 1000, 3).astype(np.float32)
        >>> y_train = np.random.randint(0, 5, (2, 1000, 1)).astype(np.int64)
        >>> x_test = np.random.rand(2, 100, 3).astype(np.float32)
        >>> knn.fit(x_train, y_train)
        >>> labels, indices = knn.predict(x_test)
    """

    def __init__(self, k: int, device_id: int = 0) -> None: ...

    def fit(
        self,
        x_train_coords: NDArray[np.float32],
        y_train: NDArray[np.int64],
    ) -> None:
        """Fit the KNN model with training data.

        Args:
            x_train_coords: Training coordinates with shape (B, N, T) where
                B is batch size, N is number of training points, and T is
                coordinate dimensions. Must be float32.
            y_train: Training labels with shape (B, N, 1). Must be int64.
        """
        ...

    def predict(
        self,
        x_test_coords: NDArray[np.float32],
        k: Optional[int] = None,
    ) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
        """Predict nearest neighbors for test points (fixed K mode).

        Args:
            x_test_coords: Test coordinates with shape (B, M, T) where
                B is batch size, M is number of test points, and T is
                coordinate dimensions. Must be float32.
            k: Number of neighbors. If None, uses the default value from
                initialization.

        Returns:
            A tuple of (labels, indices):
            - labels: Neighbor labels with shape (B, M, K). Invalid positions
              are filled with -1.
            - indices: Neighbor indices with shape (B, M, K). Invalid positions
              are filled with -1. Indices correspond to positions in the
              training set.
        """
        ...

    def predict_range(
        self,
        x_test_coords: NDArray[np.float32],
        k: int,
        distance_threshold: float,
    ) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
        """Predict neighbors within distance threshold (range-KNN mode).

        Returns neighbors that are within `distance_threshold` (L2 squared
        distance), limited to at most `k` neighbors (taking the closest ones).

        Args:
            x_test_coords: Test coordinates with shape (B, M, T) where
                B is batch size, M is number of test points, and T is
                coordinate dimensions. Must be float32.
            k: Maximum number of neighbors.
            distance_threshold: Distance threshold (L2 squared distance).
                Neighbors with squared distance greater than this value
                are excluded.

        Returns:
            A tuple of (labels, indices):
            - labels: Neighbor labels with shape (B, M, K). Invalid positions
              (no neighbor within threshold) are filled with -1.
            - indices: Neighbor indices with shape (B, M, K). Invalid positions
              are filled with -1.
        """
        ...
