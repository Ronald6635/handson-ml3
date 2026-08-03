"""
GPU Configuration Verification Module

This module provides diagnostic tools to verify that the deep learning
environment correctly detects and utilizes NVIDIA GPU hardware via 
the Keras 3 and PyTorch backend stack.

Key features:
- Backend configuration verification
- CUDA availability check
- GPU hardware metadata reporting
- Functional execution test on GPU device
"""

# CRITICAL: Backend must be set BEFORE importing ANY Keras or TensorFlow components.
import os

import keras.backend
import tensorflow
os.environ["KERAS_BACKEND"] = "torch" # Set Keras 3 backend to PyTorch

import keras # Import Keras 3 (standalone library)
import torch # Import PyTorch
# If you explicitly need TensorFlow for other non-Keras functionalities,
# import it AFTER keras. However, for Keras 3 with PyTorch backend testing,
# it's often cleaner to avoid importing tensorflow if not strictly necessary.
# import tensorflow as tf 

from typing import Dict, Any, Union, Optional
import numpy as np
from sklearn.preprocessing import StandardScaler
from datetime import datetime # Added for docstring example

# =============================================================================
# DIAGNOSTIC UTILITIES
# =============================================================================

def verify_gpu_status() -> Dict[str, Union[str, bool, float]]:
    """
    Perform a comprehensive check of the GPU environment.
    
    This function queries both the Keras configuration and the underlying 
    PyTorch CUDA state to ensure hardware acceleration is active.
    
    Returns:
        Dictionary containing:
            - 'backend' (str): The active Keras backend name.
            - 'is_cuda_active' (bool): Whether CUDA is detectable by Torch.
            - 'device_name' (str): The name of the primary GPU device.
            - 'total_memory_gb' (float): Total VRAM available in Gigabytes.
            
    Example:
        >>> results = verify_gpu_status()
        >>> if results['is_cuda_active']:
        ...     print(f"Running on {results['device_name']}")
    """
    status: Dict[str, Any] = {
        "backend": tensorflow.keras.backend.backend(), # Using standalone Keras config to report backend
        "is_cuda_active": torch.cuda.is_available(),
        "device_name": "CPU",
        "total_memory_gb": 0.0
    }

    if status["is_cuda_active"]:
        status["device_name"] = torch.cuda.get_device_name(0)
        # Convert bytes to Gigabytes for readability
        # Formula: GB = total_bytes / 1024^3
        status["total_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9

    return status

# =============================================================================
# REFINED DIAGNOSTIC EXECUTION
# =============================================================================

def execute_gpu_test(results: Dict[str, Any]) -> None:
    """
    Execute a small deep learning model to verify GPU functionality.

    This function creates a simple Keras Sequential model, compiles it,
    generates dummy data, and attempts to run a prediction.
    It prints results to the console, indicating success or failure
    and the device used.

    Args:
        results: Dictionary containing diagnostic information from
                 verify_gpu_status, especially 'is_cuda_active' and 'backend'.

    Raises:
        Exception: If the model compilation or prediction fails unexpectedly.

    Example:
        >>> diag_results = {'is_cuda_active': True, 'backend': 'torch'}
        >>> execute_gpu_test(diag_results)
        # Output will show model training/prediction info.
    """
    print("\n" + "=" * 50)
    print("Executing GPU Test (Keras with PyTorch Backend)")
    print("=" * 50)

    try:
        # Create a simple Keras model with PyTorch backend
        model = keras.Sequential([
            keras.layers.Dense(32, activation='relu', input_shape=(5,)),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        print("Keras model compiled successfully.")

        # Generate dummy data (using numpy and then converting to torch tensor)
        X_train_np = np.random.rand(100, 5).astype(np.float32)
        y_train_np = (np.random.rand(100, 1) > 0.5).astype(np.float32)
        X_test_np = np.random.rand(10, 5).astype(np.float32)

        # Scale features
        scaler = StandardScaler()
        X_train_scaled_np = scaler.fit_transform(X_train_np)
        X_test_scaled_np = scaler.transform(X_test_np)

        # Convert NumPy arrays to PyTorch tensors
        # Keras with PyTorch backend expects PyTorch tensors
        X_train = torch.from_numpy(X_train_scaled_np)
        y_train = torch.from_numpy(y_train_np)
        X_test = torch.from_numpy(X_test_scaled_np)

        # Move tensors to GPU if available and PyTorch can use CUDA
        if results['is_cuda_active']:
            device = torch.device("cuda:0")
            X_train = X_train.to(device)
            y_train = y_train.to(device)
            X_test = X_test.to(device)
            print(f"Data moved to GPU: {results['device_name']}")
        else:
            device = torch.device("cpu")
            print("Data remains on CPU.")
        
        # Train the model briefly
        print("Training Keras model...")
        model.fit(X_train, y_train, epochs=2, batch_size=32, verbose=0)
        print("Model training complete.")

        # Make a prediction
        print("Making a prediction...")
        prediction: torch.Tensor | Any = model(X_test)
        print("Prediction successful.")
        print(f"Sample prediction (first 5 values): {prediction.flatten()[:5].tolist()}")

        print("\nGPU test completed successfully.")

    except Exception as e:
        print(f"\nERROR during GPU test: {e}")
        print("Please check your Keras/PyTorch/CUDA installation and configuration.")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("GPU Setup Verification")
    print("=" * 50)
    
    diagnostic_results = verify_gpu_status()
    print(f"Keras Backend: {diagnostic_results['backend']}")
    print(f"PyTorch CUDA Available: {diagnostic_results['is_cuda_active']}")
    if diagnostic_results['is_cuda_active']:
        print(f"Primary GPU Device: {diagnostic_results['device_name']}")
        print(f"Total GPU Memory: {diagnostic_results['total_memory_gb']:.2f} GB")
    else:
        print("WARNING: CUDA is not detected. Training will be slow on CPU.")
    
    execute_gpu_test(diagnostic_results)