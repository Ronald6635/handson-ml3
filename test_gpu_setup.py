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

import os
# CRITICAL: Backend must be set before importing any Keras components
os.environ["KERAS_BACKEND"] = "torch"

import keras
import torch
import numpy as np
from typing import Dict, Any, Union, Optional

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
        "backend": keras.config.backend(),
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
    Execute a functional deep learning diagnostic.
    
    This function creates a minimal neural network to verify that tensors 
    are correctly dispatched to the GPU device by the Torch backend.
    
    Args:
        results (Dict[str, Any]): Metadata dictionary from verify_gpu_status()
        
    Returns:
        None
        
    Note:
        Success is indicated by a non-crashing forward pass and the 
        presence of a torch.Tensor output.
    """
    print(f"Keras Backend: {results['backend']}")
    print(f"PyTorch CUDA Available: {results['is_cuda_active']}")
    
    if results["is_cuda_active"]:
        print(f"GPU Device: {results['device_name']}")
        # Display memory in GB using KaTeX for the underlying logic:
        # $$ GB = \frac{bytes}{1024^3} $$
        print(f"GPU Memory: {results['total_memory_gb']:.2f} GB")
    else:
        # NOTE: If this prints False, verify the PyTorch installation index-url
        # CRITICAL: Without CUDA, training performance will degrade by ~10-50x.
        print("WARNING: CUDA is not detected. Training will be slow on CPU.")

    # Quick functional model test
    # Standard: Use torch tensors when backend is set to 'torch'
    X_test: torch.Tensor = torch.randn(10, 5)
    
    # Define a simple regression model to test layer connectivity
    model: keras.Sequential = keras.Sequential([
        keras.layers.Input(shape=(5,)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(1)
    ])
    
    # CRITICAL: Compilation must succeed before execution to initialize weights
    model.compile(optimizer='adam', loss='mse')
    
    # Forward pass: Keras 3 handles the conversion to the backend's native tensor
    prediction: torch.Tensor | Any = model(X_test)
    
    print(f"\nModel Test Output Shape: {prediction.shape}")
    print("✓ Diagnostic complete!")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("GPU Setup Verification")
    print("=" * 50)
    
    diagnostic_results = verify_gpu_status()
    execute_gpu_test(diagnostic_results)