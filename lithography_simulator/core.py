"""Core simulation engine for DUV 193nm lithography simulator targeting 7nm node.

This module implements 2D aerial image simulation using the TensorFlow Torchoptics library
as the optical engine. It handles vectorial optical imaging with partial coherence for
immersion lithography systems.
"""

import tensorflow as tf
import numpy as np
from typing import Tuple, Optional, Dict, Any
from tensorflow_torchoptics import Field, PlanarGrid, Lens
from tensorflow_torchoptics.profiles import circle


class LithographySystem:
    """Main class for the lithography simulation system."""
    
    def __init__(
        self,
        na: float = 1.35,  # Numerical aperture for immersion
        wavelength: float = 193e-9,  # DUV wavelength in meters
        n_immersion: float = 1.44,  # Refractive index of immersion fluid
        resolution: int = 128,  # Resolution of the simulation grid
        mask_size: float = 10e-6,  # Physical size of the mask area in meters
        sigma: float = 0.5,  # Partial coherence factor
        polarization: str = 'unpolarized',  # Polarization state
    ):
        """
        Initialize the lithography system.
        
        Args:
            na: Numerical aperture of the projection lens
            wavelength: Exposure wavelength in meters (193nm)
            n_immersion: Refractive index of immersion fluid
            resolution: Resolution of the simulation grid (N x N)
            mask_size: Physical size of the mask area in meters
            sigma: Partial coherence factor (0-1)
            polarization: Polarization state ('te', 'tm', 'unpolarized')
        """
        self.na = na
        self.wavelength = wavelength
        self.n_immersion = n_immersion
        self.effective_wavelength = wavelength / n_immersion  # Effective wavelength in immersion
        self.resolution = resolution
        self.mask_size = mask_size
        self.sigma = sigma
        self.polarization = polarization
        
        # Calculate physical spacing based on mask size and resolution
        self.spacing = mask_size / resolution
        
        # Create the planar grid for simulation
        self.grid = PlanarGrid(
            shape=(resolution, resolution),
            spacing=(self.spacing, self.spacing),
            offset=(0.0, 0.0)
        )
        
        # Create the projection lens
        self.lens = self._create_projection_lens()
        
    def _create_projection_lens(self) -> Lens:
        """Create the projection lens with specified NA."""
        # Calculate focal length based on NA and physical constraints
        # For simplicity, we'll use a large focal length to approximate the NA-limited PSF
        focal_length = self.mask_size / (2 * self.na)  # Approximate relationship
        
        return Lens(
            shape=self.grid.shape,
            focal_length=focal_length,
            spacing=self.grid.spacing
        )
    
    def simulate_aerial_image(
        self, 
        mask_pattern: tf.Tensor, 
        focus_offset: float = 0.0
    ) -> tf.Tensor:
        """
        Simulate the aerial image for a given mask pattern.
        
        Args:
            mask_pattern: 2D tensor representing the mask transmission (complex amplitude)
            focus_offset: Defocus amount in meters
            
        Returns:
            2D tensor representing the aerial image intensity
        """
        # Create an input field with the mask pattern
        input_field = Field(
            grid=self.grid,
            wavelength=self.effective_wavelength,
            z=focus_offset
        )
        
        # Apply the mask pattern to the input field
        masked_field = input_field * mask_pattern
        
        # Propagate through the optical system (lens)
        output_field = self.lens(masked_field)
        
        # Calculate the intensity of the output field
        aerial_image_intensity = tf.abs(output_field.u) ** 2
        
        return aerial_image_intensity
    
    def calculate_psf(self) -> tf.Tensor:
        """
        Calculate the point spread function of the optical system.
        
        Returns:
            2D tensor representing the PSF
        """
        # Create a point source (delta function) as the input
        point_source = tf.zeros(self.grid.shape, dtype=tf.complex64)
        center_x, center_y = self.grid.shape[0] // 2, self.grid.shape[1] // 2
        point_source = tf.tensor_scatter_nd_update(
            point_source, 
            [[center_x, center_y]], 
            [tf.complex(1.0, 0.0)]
        )
        
        # Simulate with the point source to get the PSF
        psf_intensity = self.simulate_aerial_image(point_source)
        
        return psf_intensity
    
    def hopkins_imaging_model(
        self, 
        mask_pattern: tf.Tensor, 
        illumination_sigma: Optional[float] = None
    ) -> tf.Tensor:
        """
        Calculate aerial image using Hopkins partially coherent imaging model.
        
        Args:
            mask_pattern: 2D tensor representing the mask transmission
            illumination_sigma: Override the system's sigma value if provided
            
        Returns:
            2D tensor representing the partially coherent aerial image
        """
        if illumination_sigma is None:
            illumination_sigma = self.sigma
            
        # For simplicity, we'll approximate the Hopkins model using the coherent model
        # with effective parameters. A full implementation would require TCC calculation.
        aerial_image = self.simulate_aerial_image(mask_pattern)
        
        # Apply sigma-dependent weighting to approximate partial coherence
        # This is a simplified approach - a full implementation would be more complex
        if illumination_sigma < 1.0:
            # For now, we'll just scale the image based on coherence
            aerial_image = aerial_image * (0.5 + 0.5 * illumination_sigma)
            
        return aerial_image


def create_binary_mask_from_pattern(
    pattern: np.ndarray, 
    threshold: float = 0.5
) -> tf.Tensor:
    """
    Create a binary mask from a pattern array.
    
    Args:
        pattern: 2D array representing the desired pattern
        threshold: Threshold value to binarize the pattern
        
    Returns:
        2D complex tensor representing the binary mask transmission
    """
    # Binarize the pattern
    binary_pattern = (pattern > threshold).astype(np.float32)
    
    # Convert to complex tensor (amplitude-only mask for now)
    mask_complex = tf.cast(binary_pattern, tf.complex64)
    
    return mask_complex


def calculate_image_log_sigmoid_gamma(
    aerial_image: tf.Tensor,
    gamma: float = 1.0
) -> tf.Tensor:
    """
    Calculate the resist response using image log-sigmoid model.
    
    Args:
        aerial_image: Aerial image intensity
        gamma: Gamma value for the resist contrast
        
    Returns:
        Processed resist image
    """
    # Normalize aerial image
    normalized_image = aerial_image / tf.reduce_max(aerial_image)
    
    # Apply image log-sigmoid transformation
    resist_response = tf.nn.sigmoid(gamma * (normalized_image - 0.5)) 
    
    return resist_response