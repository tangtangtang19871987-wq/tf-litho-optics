"""Mask module for DUV 193nm lithography simulator.

This module handles 2D mask representation and manipulation for the 7nm node lithography simulator.
It includes binary masks, phase shift masks, and mask enhancement techniques.
"""

import tensorflow as tf
import numpy as np
from typing import Tuple, Optional, Union
from scipy import ndimage
from skimage import draw, morphology


class Mask:
    """Base class for mask representation in lithography."""
    
    def __init__(self, shape: Tuple[int, int], pixel_size: float = 1e-9):
        """
        Initialize the mask.
        
        Args:
            shape: Shape of the mask (height, width)
            pixel_size: Physical size of each pixel in meters
        """
        self.shape = shape
        self.pixel_size = pixel_size
        self.pattern = tf.zeros(shape, dtype=tf.complex64)
        
    def set_pattern(self, pattern: tf.Tensor):
        """Set the mask pattern."""
        if pattern.shape != self.shape:
            raise ValueError(f"Pattern shape {pattern.shape} does not match mask shape {self.shape}")
        self.pattern = pattern
        
    def get_transmission(self) -> tf.Tensor:
        """Get the transmission function of the mask."""
        return self.pattern


class BinaryMask(Mask):
    """Binary amplitude mask for DUV lithography."""
    
    def __init__(self, shape: Tuple[int, int], pixel_size: float = 1e-9):
        super().__init__(shape, pixel_size)
        
    def create_from_binary_pattern(self, binary_pattern: np.ndarray) -> 'BinaryMask':
        """
        Create a binary mask from a binary pattern.
        
        Args:
            binary_pattern: 2D numpy array with 0 (opaque) and 1 (transmissive) values
            
        Returns:
            Self for method chaining
        """
        if binary_pattern.shape != self.shape:
            raise ValueError(f"Pattern shape {binary_pattern.shape} does not match mask shape {self.shape}")
            
        # Convert to complex tensor (real part represents transmission)
        complex_pattern = tf.cast(binary_pattern, tf.complex64)
        self.pattern = complex_pattern
        return self
        
    def add_assist_features(self, feature_size: int = 2) -> 'BinaryMask':
        """
        Add sub-resolution assist features (SRAF) to improve imaging.
        
        Args:
            feature_size: Size of assist features in pixels
            
        Returns:
            Self for method chaining
        """
        # Convert to numpy for morphological operations
        pattern_real = tf.math.real(self.pattern).numpy()
        
        # Find edges in the pattern
        selem = morphology.disk(feature_size)
        eroded = ndimage.binary_erosion(pattern_real, structure=selem)
        edges = pattern_real - eroded
        
        # Add assist features near edges
        assist_locations = ndimage.binary_dilation(edges, structure=morphology.disk(3*feature_size))
        assist_locations = assist_locations.astype(bool) & ~pattern_real.astype(bool)
        
        # Add small rectangles as assist features
        assist_pattern = pattern_real.copy()
        assist_pattern[assist_locations] = 1.0
        
        # Update the complex pattern
        self.pattern = tf.cast(assist_pattern, tf.complex64)
        return self


class PhaseShiftMask(Mask):
    """Phase shift mask for improved resolution in DUV lithography."""
    
    def __init__(self, shape: Tuple[int, int], pixel_size: float = 1e-9, phase_shift: float = np.pi):
        super().__init__(shape, pixel_size)
        self.phase_shift = phase_shift  # Usually π for alternating PSM
        
    def create_alternating_psm(self, binary_pattern: np.ndarray) -> 'PhaseShiftMask':
        """
        Create an alternating phase shift mask.
        
        Args:
            binary_pattern: 2D numpy array with 0 (opaque) and 1 (transmissive) values
            
        Returns:
            Self for method chaining
        """
        if binary_pattern.shape != self.shape:
            raise ValueError(f"Pattern shape {binary_pattern.shape} does not match mask shape {self.shape}")
        
        # Find connected components to alternate phases
        labeled, num_features = ndimage.label(binary_pattern)
        
        # Create phase pattern (0 phase for odd-numbered regions, pi phase for even-numbered)
        phase_pattern = np.zeros(self.shape, dtype=np.float32)
        for i in range(1, num_features + 1):
            region_mask = (labeled == i)
            if i % 2 == 0:  # Even-numbered regions get phase shift
                phase_pattern[region_mask] = self.phase_shift
        
        # Create complex transmission (amplitude=1 for transmissive, 0 for opaque)
        amplitude_pattern = binary_pattern.astype(np.complex64)
        phase_shift_complex = np.exp(1j * phase_pattern).astype(np.complex64)
        
        # Combine amplitude and phase
        complex_pattern = amplitude_pattern * phase_shift_complex
        self.pattern = tf.constant(complex_pattern, dtype=tf.complex64)
        
        return self


def create_line_space_pattern(width: int, height: int, line_width: int, pitch: int) -> np.ndarray:
    """
    Create a line-space pattern for testing.
    
    Args:
        width: Width of the pattern
        height: Height of the pattern
        line_width: Width of each line
        pitch: Pitch between lines (line + space)
        
    Returns:
        2D binary array representing the line-space pattern
    """
    pattern = np.zeros((height, width), dtype=np.float32)
    
    # Calculate how many periods fit in the width
    num_periods = width // pitch
    
    for i in range(num_periods):
        start_x = i * pitch
        end_x = min(start_x + line_width, width)
        pattern[:, start_x:end_x] = 1.0
        
    return pattern


def create_contact_hole_pattern(width: int, height: int, hole_size: int, pitch: int) -> np.ndarray:
    """
    Create a contact hole array pattern for testing.
    
    Args:
        width: Width of the pattern
        height: Height of the pattern
        hole_size: Size of each contact hole
        pitch: Pitch between holes
        
    Returns:
        2D binary array representing the contact hole pattern
    """
    pattern = np.ones((height, width), dtype=np.float32)
    
    # Calculate how many periods fit in each dimension
    num_x_periods = width // pitch
    num_y_periods = height // pitch
    
    for i in range(num_x_periods):
        for j in range(num_y_periods):
            center_x = i * pitch + pitch // 2
            center_y = j * pitch + pitch // 2
            
            # Draw a square hole
            start_x = max(center_x - hole_size // 2, 0)
            end_x = min(center_x + hole_size // 2, width)
            start_y = max(center_y - hole_size // 2, 0)
            end_y = min(center_y + hole_size // 2, height)
            
            pattern[start_y:end_y, start_x:end_x] = 0.0
            
    return pattern


def mask_fracture(mask_pattern: np.ndarray, max_shape_size: int = 100) -> list:
    """
    Perform mask fracturing to break complex shapes into simpler ones.
    
    Args:
        mask_pattern: 2D binary array representing the mask pattern
        max_shape_size: Maximum size of individual fractured shapes
        
    Returns:
        List of fractured shapes as coordinate arrays
    """
    # Label connected components
    labeled, num_features = ndimage.label(mask_pattern)
    
    fractured_shapes = []
    for i in range(1, num_features + 1):
        # Get coordinates of this feature
        coords = np.where(labeled == i)
        coords_array = np.column_stack(coords)
        
        # If the feature is too large, subdivide it
        if len(coords_array) > max_shape_size:
            # For now, just return the original shape; in practice, more sophisticated
            # subdivision algorithms would be used
            fractured_shapes.append(coords_array)
        else:
            fractured_shapes.append(coords_array)
    
    return fractured_shapes