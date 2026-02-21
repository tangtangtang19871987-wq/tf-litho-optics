"""Illumination module for DUV 193nm lithography simulator.

This module handles various illumination conditions for immersion lithography
systems targeting the 7nm node. It includes dipole, quadrupole, annular,
and custom illumination modes.
"""

import tensorflow as tf
import numpy as np
from typing import Tuple, Optional
from scipy.special import jv


class IlluminationSource:
    """Base class for illumination source representation."""
    
    def __init__(self, shape: Tuple[int, int], wavelength: float = 193e-9):
        """
        Initialize the illumination source.
        
        Args:
            shape: Shape of the pupil grid (height, width)
            wavelength: Wavelength of illumination in meters
        """
        self.shape = shape
        self.wavelength = wavelength
        self.source_map = tf.ones(shape, dtype=tf.float32)  # Uniform by default
        
    def get_source_map(self) -> tf.Tensor:
        """Get the illumination source map."""
        return self.source_map


class CoherentSource(IlluminationSource):
    """Coherent illumination source."""
    
    def __init__(self, shape: Tuple[int, int], wavelength: float = 193e-9):
        super().__init__(shape, wavelength)
        # For coherent illumination, the source is essentially a delta function
        # But for practical simulation, we use uniform illumination
        self.source_map = tf.ones(shape, dtype=tf.float32)


class PartialCoherentSource(IlluminationSource):
    """Partially coherent illumination with adjustable sigma."""
    
    def __init__(
        self, 
        shape: Tuple[int, int], 
        sigma: float = 0.5, 
        wavelength: float = 193e-9,
        source_type: str = 'annular'
    ):
        """
        Initialize partially coherent illumination.
        
        Args:
            shape: Shape of the pupil grid (height, width)
            sigma: Coherence factor (0-1)
            wavelength: Wavelength of illumination in meters
            source_type: Type of illumination ('annular', 'quasar', 'dipole')
        """
        super().__init__(shape, wavelength)
        self.sigma = sigma
        self.source_type = source_type
        self.source_map = self._create_source_map()
        
    def _create_source_map(self) -> tf.Tensor:
        """Create the source map based on the source type and sigma."""
        h, w = self.shape
        y, x = tf.meshgrid(tf.range(h, dtype=tf.float32), tf.range(w, dtype=tf.float32), indexing='ij')
        
        # Center the coordinates
        center_y, center_x = h / 2.0, w / 2.0
        y = y - center_y
        x = x - center_x
        
        # Normalize coordinates to unit circle
        rho = tf.sqrt(x**2 + y**2) / tf.reduce_max([h, w]) * 2
        
        if self.source_type == 'annular':
            # Annular illumination: ring-shaped source
            inner_radius = self.sigma * 0.7
            outer_radius = self.sigma * 1.0
            source_map = tf.cast(
                (rho >= inner_radius) & (rho <= outer_radius), 
                tf.float32
            )
            
        elif self.source_type == 'quasar':
            # Quasar illumination: four poles
            # Define four quadrants with different intensities
            source_map = tf.zeros_like(rho)
            
            # Define the four poles (top-left, top-right, bottom-left, bottom-right)
            angle = tf.atan2(y, x)
            radius_condition = (rho >= self.sigma * 0.6) & (rho <= self.sigma * 1.0)
            
            # Top-left quadrant
            source_map += tf.cast(
                radius_condition & (angle >= 0) & (angle < np.pi/2), 
                tf.float32
            ) * 0.5
            # Top-right quadrant
            source_map += tf.cast(
                radius_condition & (angle >= np.pi/2) & (angle < np.pi), 
                tf.float32
            ) * 0.5
            # Bottom-left quadrant
            source_map += tf.cast(
                radius_condition & (angle >= -np.pi) & (angle < -np.pi/2), 
                tf.float32
            ) * 0.5
            # Bottom-right quadrant
            source_map += tf.cast(
                radius_condition & (angle >= -np.pi/2) & (angle < 0), 
                tf.float32
            ) * 0.5
            
        elif self.source_type == 'dipole':
            # Dipole illumination: two poles
            source_map = tf.zeros_like(rho)
            
            # Define two poles (left and right)
            angle = tf.atan2(y, x)
            radius_condition = (rho >= self.sigma * 0.6) & (rho <= self.sigma * 1.0)
            
            # Left pole
            source_map += tf.cast(
                radius_condition & (angle >= -np.pi/4) & (angle <= np.pi/4), 
                tf.float32
            )
            # Right pole
            source_map += tf.cast(
                radius_condition & ((angle >= 3*np.pi/4) | (angle <= -3*np.pi/4)), 
                tf.float32
            )
            
        else:
            # Default to uniform if unknown type
            source_map = tf.ones_like(rho) * self.sigma
            
        return source_map


class VectorialPolarizationSource(IlluminationSource):
    """Vectorial illumination source with polarization control."""
    
    def __init__(
        self, 
        shape: Tuple[int, int], 
        polarization: str = 'unpolarized',
        wavelength: float = 193e-9
    ):
        """
        Initialize vectorial illumination with polarization control.
        
        Args:
            shape: Shape of the pupil grid (height, width)
            polarization: Polarization state ('te', 'tm', 'unpolarized', 'radial', 'azimuthal')
            wavelength: Wavelength of illumination in meters
        """
        super().__init__(shape, wavelength)
        self.polarization = polarization
        self.source_map = self._create_polarization_map()
        
    def _create_polarization_map(self) -> tf.Tensor:
        """Create the polarization-dependent source map."""
        h, w = self.shape
        y, x = tf.meshgrid(tf.range(h, dtype=tf.float32), tf.range(w, dtype=tf.float32), indexing='ij')
        
        # Center the coordinates
        center_y, center_x = h / 2.0, w / 2.0
        y = y - center_y
        x = x - center_x
        
        # Calculate radial and angular coordinates
        rho = tf.sqrt(x**2 + y**2)
        phi = tf.atan2(y, x)
        
        # Create the basic source map (uniform for now)
        base_source = tf.ones_like(rho)
        
        if self.polarization == 'te':
            # Transverse electric (TE) polarization - electric field perpendicular to plane of incidence
            # For this simulation, we weight by the TE transmission efficiency
            # Simplified model: TE has uniform response
            return base_source
        elif self.polarization == 'tm':
            # Transverse magnetic (TM) polarization - electric field parallel to plane of incidence
            # TM has stronger response at higher angles
            # Simplified model: enhanced at edges
            normalized_rho = rho / tf.reduce_max(rho)
            tm_weight = 1.0 + 0.5 * normalized_rho  # Enhanced at edges
            return base_source * tm_weight
        elif self.polarization == 'radial':
            # Radial polarization - electric field oriented radially
            # Simplified model: varies with angle
            return base_source
        elif self.polarization == 'azimuthal':
            # Azimuthal polarization - electric field oriented tangentially
            # Simplified model: varies with angle
            return base_source
        else:  # unpolarized
            # Average of TE and TM effects
            normalized_rho = rho / tf.reduce_max(rho)
            avg_weight = 1.0 + 0.25 * normalized_rho  # Mild enhancement at edges
            return base_source * avg_weight


def create_illumination(
    shape: Tuple[int, int],
    source_type: str = 'partial_coherent',
    **kwargs
) -> IlluminationSource:
    """
    Factory function to create different types of illumination sources.
    
    Args:
        shape: Shape of the pupil grid
        source_type: Type of illumination ('coherent', 'partial_coherent', 'vectorial')
        **kwargs: Additional arguments for specific source types
        
    Returns:
        Instance of the requested illumination source
    """
    if source_type == 'coherent':
        return CoherentSource(shape, kwargs.get('wavelength', 193e-9))
    elif source_type == 'partial_coherent':
        return PartialCoherentSource(
            shape, 
            kwargs.get('sigma', 0.5),
            kwargs.get('wavelength', 193e-9),
            kwargs.get('source_type', 'annular')
        )
    elif source_type == 'vectorial':
        return VectorialPolarizationSource(
            shape,
            kwargs.get('polarization', 'unpolarized'),
            kwargs.get('wavelength', 193e-9)
        )
    else:
        raise ValueError(f"Unknown source type: {source_type}")


def combine_illumination_sources(
    source1: IlluminationSource, 
    source2: IlluminationSource, 
    weight1: float = 0.5
) -> tf.Tensor:
    """
    Combine two illumination sources with specified weights.
    
    Args:
        source1: First illumination source
        source2: Second illumination source
        weight1: Weight for the first source (weight2 = 1 - weight1)
        
    Returns:
        Combined illumination source map
    """
    weight2 = 1.0 - weight1
    combined_map = weight1 * source1.get_source_map() + weight2 * source2.get_source_map()
    
    # Normalize to maintain energy conservation
    combined_map = combined_map / tf.reduce_max(combined_map)
    
    return combined_map