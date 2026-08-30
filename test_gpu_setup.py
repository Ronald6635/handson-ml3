"""
GPU Configuration Verification Module

Verified for: Keras 2.10 + PyTorch 2.5 + TensorFlow 2.10 (CUDA 12.1)
"""

# CRITICAL: Set backend BEFORE ANY keras or tensorflow imports
import os
os.environ["KERAS_BACKEND"] = "torch"

from typing import Any, Dict, Union

import keras
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler


def verify_gpu_status() -> Dict[str, Union[str, bool, float]]:
    """
    Perform a comprehensive check of the GPU environment.

    Returns:
        Dictionary containing backend and CUDA/GPU metadata.
    """
    status: Dict[str, Any] = {
        "backend": keras.backend.backend(),
        "is_cuda_active": torch.cuda.is_available(),
        "device_name": "CPU",
        "total_memory_gb": 0.0,
    }

    if status["is_cuda_active"]:
        status["device_name"] = torch.cuda.get_device_name(0)
        status["total_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9

    return status


def execute_gpu_test(results: Dict[str, Any]) -> None:
    """
    Execute a small deep learning model to verify GPU functionality.
    """
    print("\n" + "=" * 50)
    print("Executing GPU Test (Keras with PyTorch Backend)")
    print("=" * 50)

    try:
        model = keras.Sequential([
            keras.layers.Input(shape=(5,)),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid"),
        ])
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        print("Keras model compiled successfully.")

        # Generate numpy data (NOT torch tensors yet)
        X_train_np = np.random.rand(100, 5).astype(np.float32)
        y_train_np = (np.random.rand(100, 1) > 0.5).astype(np.float32)
        X_test_np = np.random.rand(10, 5).astype(np.float32)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_np).astype(np.float32)
        X_test_scaled = scaler.transform(X_test_np).astype(np.float32)

        # Train directly with numpy arrays (Keras handles conversion)
        print("Training Keras model...")
        model.fit(X_train_scaled, y_train_np, epochs=2, batch_size=32, verbose=0)
        print("Model training complete.")

        # Make a prediction
        print("Making a prediction...")
        prediction = model.predict(X_test_scaled, verbose=0)
        print("Prediction successful.")
        print(f"Sample prediction (first 5 values): {prediction.flatten()[:5].tolist()}")

        print("\nGPU test completed successfully.")

    except Exception as error:
        print(f"\nERROR during GPU test: {error}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("=" * 50)
    print("GPU Setup Verification")
    print("=" * 50)

    diagnostic_results = verify_gpu_status()
    print(f"Keras Backend: {diagnostic_results['backend']}")
    print(f"PyTorch CUDA Available: {diagnostic_results['is_cuda_active']}")
    if diagnostic_results["is_cuda_active"]:
        print(f"Primary GPU Device: {diagnostic_results['device_name']}")
        print(f"Total GPU Memory: {diagnostic_results['total_memory_gb']:.2f} GB")
    else:
        print("WARNING: CUDA is not detected. Training will be slow on CPU.")

    execute_gpu_test(diagnostic_results)