"""GPU-accelerated batch point cloud KNN classifier.

This module provides a high-performance KNN implementation that supports:
- Fixed K nearest neighbors
- Range + count limited nearest neighbors (range-KNN)
"""

from rust_gpu_knn.rust_gpu_knn import GPUBatchPointCloudKNN

__all__ = ["GPUBatchPointCloudKNN"]
__version__ = "0.1.0"
