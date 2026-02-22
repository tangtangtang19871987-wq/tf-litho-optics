"""
Data Generation Module for Lithography Simulator

This module contains functions to generate training data for ML models
that predict aerial images from mask layouts.
"""

import numpy as np
import tensorflow as tf
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core import simulate_aerial_image
from mask import generate_random_mask
from illumination import PartialCoherentSource
from optics import OpticalSystem


def generate_training_data(
    num_samples=100,
    resolution=8e-9,  # 8nm resolution
    dimension=256,    # 256x256 pixels
    pitch=75e-9,      # 75nm pitch
    wavelength=193e-9,  # DUV 193nm
    immersion=True,
    na=1.35,  # Numerical aperture for immersion lithography
    sigma_in=0.6,  # Inner sigma for dipole illumination
    sigma_out=0.85  # Outer sigma for dipole illumination
):
    """
    Generate training data for ML models.
    
    Args:
        num_samples: Number of samples to generate
        resolution: Resolution in meters
        dimension: Dimension of square images (dimension x dimension)
        pitch: Pitch of the periodic pattern in meters
        wavelength: Wavelength in meters
        immersion: Whether using immersion lithography
        na: Numerical aperture
        sigma_in: Inner sigma for illumination
        sigma_out: Outer sigma for illumination
        
    Returns:
        tuple: (masks, aerial_images) as numpy arrays
    """
    # Calculate physical dimensions
    physical_size = resolution * dimension  # Size in meters
    
    # Create illumination source
    illumination = PartialCoherentSource(
        shape=(dimension, dimension),
        wavelength=wavelength,
        sigma=sigma_out,  # Using sigma_out as the primary sigma
        source_type='dipole'
    )
    
    # Create optical system
    optics = OpticalSystem(
        wavelength=wavelength,
        na=na,
        resolution=resolution
    )
    
    # Initialize storage for data
    masks = np.zeros((num_samples, dimension, dimension), dtype=np.float32)
    aerial_images = np.zeros((num_samples, dimension, dimension), dtype=np.float32)
    
    print(f"Generating {num_samples} training samples...")
    
    for i in range(num_samples):
        if i % 10 == 0:
            print(f"Generated {i}/{num_samples} samples")
            
        # Generate random mask pattern
        # We'll create a metal layer-like pattern with features around the pitch size
        mask = generate_random_mask(
            shape=(dimension, dimension),
            resolution=resolution,
            feature_size=pitch,
            pattern_type='metal_layer'
        )
        
        # Simulate aerial image
        aerial_image = simulate_aerial_image(
            mask=mask,
            illumination=illumination,
            optics=optics,
            resist_thickness=0.1e-6  # 100nm resist thickness
        )
        
        # Store the data
        masks[i] = mask
        aerial_images[i] = aerial_image
    
    print(f"Completed generating {num_samples} training samples.")
    
    return masks, aerial_images


def save_training_data(masks, aerial_images, filepath='training_data.npz'):
    """
    Save training data to a compressed numpy file.
    
    Args:
        masks: Array of mask patterns
        aerial_images: Array of corresponding aerial images
        filepath: Path to save the data
    """
    np.savez_compressed(filepath, X=masks, Y=aerial_images)
    print(f"Training data saved to {filepath}")


if __name__ == "__main__":
    # Generate training data with the specified parameters
    masks, aerial_images = generate_training_data(
        num_samples=100,
        resolution=8e-9,    # 8nm resolution
        dimension=256,      # 256x256 images
        pitch=75e-9,        # 75nm pitch
        wavelength=193e-9,  # DUV 193nm
        immersion=True,
        na=1.35
    )
    
    # Save the generated data
    save_training_data(masks, aerial_images, 'duv_litho_training_data.npz')
    
    print("Data generation completed!")
    print(f"Masks shape: {masks.shape}")
    print(f"Aerial images shape: {aerial_images.shape}")
    print(f"Mask value range: [{masks.min():.3f}, {masks.max():.3f}]")
    print(f"Aerial image value range: [{aerial_images.min():.3f}, {aerial_images.max():.3f}]")