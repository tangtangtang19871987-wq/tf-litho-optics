"""Optics module for DUV 193nm lithography simulator.

This module handles the optical system of the immersion lithography tool,
including the projection lens, aberrations, and vectorial imaging effects.
"""

import tensorflow as tf
import numpy as np
from typing import Tuple, Optional, Dict, Any
from scipy.ndimage import gaussian_filter
from lithography_simulator.core import LithographySystem
from lithography_simulator.illumination import IlluminationSource


class ProjectionLens:
    """Model of the projection lens system in immersion lithography."""
    
    def __init__(
        self,
        na: float = 1.35,  # Numerical aperture
        wavelength: float = 193e-9,  # Wavelength in meters
        n_immersion: float = 1.44,  # Refractive index of immersion fluid
        magnification: float = 4.0,  # Typical reduction magnification
        shape: Tuple[int, int] = (128, 128),
        padding_factor: int = 2  # Padding for FFT operations
    ):
        """
        Initialize the projection lens model.
        
        Args:
            na: Numerical aperture of the lens
            wavelength: Exposure wavelength in meters
            n_immersion: Refractive index of immersion fluid
            magnification: Reduction magnification (typically 4x)
            shape: Shape of the simulation grid
            padding_factor: Factor for zero-padding FFT operations
        """
        self.na = na
        self.wavelength = wavelength
        self.n_immersion = n_immersion
        self.magnification = magnification
        self.shape = shape
        self.padding_factor = padding_factor
        
        # Calculate effective wavelength in immersion medium
        self.effective_wavelength = wavelength / n_immersion
        
        # Calculate maximum spatial frequency supported by the lens
        self.max_frequency = 2 * na / self.effective_wavelength
        
        # Generate pupil function
        self.pupil_function = self._generate_pupil_function()
        
    def _generate_pupil_function(self) -> tf.Tensor:
        """Generate the pupil function of the lens."""
        h, w = self.shape
        
        # Create coordinate grids
        y, x = tf.meshgrid(
            tf.range(h, dtype=tf.float32), 
            tf.range(w, dtype=tf.float32), 
            indexing='ij'
        )
        
        # Center the coordinates
        center_y, center_x = h / 2.0, w / 2.0
        y = y - center_y
        x = x - center_x
        
        # Normalize coordinates to units of lambda/NA
        scale_factor = 2 * np.pi * self.na / self.effective_wavelength
        fx = x / (w * self.effective_wavelength / (2 * self.na))
        fy = y / (h * self.effective_wavelength / (2 * self.na))
        
        # Calculate radial frequency
        rho = tf.sqrt(fx**2 + fy**2)
        
        # Create circular pupil (frequency cutoff)
        pupil = tf.cast(rho <= 1.0, tf.complex64)
        
        return pupil
    
    def apply_aberrations(
        self, 
        pupil_function: tf.Tensor, 
        zernike_coefficients: Optional[Dict[str, float]] = None
    ) -> tf.Tensor:
        """
        Apply optical aberrations to the pupil function using Zernike polynomials.
        
        Args:
            pupil_function: Original pupil function
            zernike_coefficients: Dictionary mapping Zernike terms to coefficients
            
        Returns:
            Aberrated pupil function
        """
        if zernike_coefficients is None:
            return pupil_function
            
        h, w = self.shape
        
        # Create coordinate grids normalized to unit circle
        y, x = tf.meshgrid(
            tf.range(h, dtype=tf.float32), 
            tf.range(w, dtype=tf.float32), 
            indexing='ij'
        )
        
        # Center and normalize coordinates
        center_y, center_x = h / 2.0, w / 2.0
        x = (x - center_x) / center_x
        y = (y - center_y) / center_y
        
        # Calculate radial distance and angle
        rho = tf.sqrt(x**2 + y**2)
        theta = tf.atan2(y, x)
        
        # Start with phase = 0
        phase = tf.zeros_like(rho)
        
        # Apply various Zernike aberrations
        for aberration, coefficient in zernike_coefficients.items():
            if aberration == 'defocus':
                # Zernike Z(2,0) - defocus
                phase += coefficient * (2*rho**2 - 1)
            elif aberration == 'astigmatism_x':
                # Zernike Z(2,2) - astigmatism (0°)
                phase += coefficient * rho**2 * tf.cos(2*theta)
            elif aberration == 'astigmatism_y':
                # Zernike Z(2,2) - astigmatism (45°)
                phase += coefficient * rho**2 * tf.sin(2*theta)
            elif aberration == 'coma_x':
                # Zernike Z(3,1) - coma
                phase += coefficient * rho * (3*rho**2 - 2) * tf.cos(theta)
            elif aberration == 'coma_y':
                # Zernike Z(3,1) - coma
                phase += coefficient * rho * (3*rho**2 - 2) * tf.sin(theta)
            elif aberration == 'spherical':
                # Zernike Z(4,0) - spherical aberration
                phase += coefficient * (6*rho**4 - 6*rho**2 + 1)
        
        # Apply the aberration phase to the pupil function
        aberrated_pupil = pupil_function * tf.exp(tf.complex(tf.zeros_like(phase), phase))
        
        return aberrated_pupil
    
    def calculate_psf(self, aberrations: Optional[Dict[str, float]] = None) -> tf.Tensor:
        """
        Calculate the point spread function of the lens.
        
        Args:
            aberrations: Dictionary of Zernike aberration coefficients
            
        Returns:
            Point spread function as a 2D tensor
        """
        # Get the pupil function with aberrations
        pupil_with_aberrations = self.apply_aberrations(self.pupil_function, aberrations)
        
        # Calculate PSF via FFT of the pupil function
        # Zero-pad to increase resolution
        padded_size = [s * self.padding_factor for s in self.shape]
        pad_sizes = [(0, padded_size[i] - self.shape[i]) for i in range(2)]
        padded_pupil = tf.pad(pupil_with_aberrations, pad_sizes, mode='CONSTANT')
        
        # Compute the PSF via FFT
        pupil_fft = tf.signal.fft2d(padded_pupil)
        psf_intensity = tf.abs(pupil_fft) ** 2
        
        # Crop back to original size
        psf_intensity = psf_intensity[:self.shape[0], :self.shape[1]]
        
        # Normalize
        psf_intensity = psf_intensity / tf.reduce_sum(psf_intensity)
        
        return psf_intensity


class VectorialImagingModel:
    """Model for vectorial imaging effects in high-NA immersion systems."""
    
    def __init__(
        self,
        na: float = 1.35,
        wavelength: float = 193e-9,
        n_immersion: float = 1.44,
        shape: Tuple[int, int] = (128, 128)
    ):
        """
        Initialize the vectorial imaging model.
        
        Args:
            na: Numerical aperture
            wavelength: Exposure wavelength
            n_immersion: Refractive index of immersion fluid
            shape: Shape of the simulation grid
        """
        self.na = na
        self.wavelength = wavelength
        self.n_immersion = n_immersion
        self.shape = shape
        self.k0 = 2 * np.pi / (wavelength / n_immersion)  # Wave number in immersion medium
        
    def calculate_vectorial_psf(self, polarization: str = 'unpolarized') -> tf.Tensor:
        """
        Calculate the vectorial point spread function.
        
        Args:
            polarization: Polarization state ('te', 'tm', 'unpolarized', 'radial', 'azimuthal')
            
        Returns:
            Vectorial PSF components as a dictionary
        """
        h, w = self.shape
        
        # Create coordinate grids in the focal plane
        y, x = tf.meshgrid(
            tf.range(h, dtype=tf.float32) - h/2, 
            tf.range(w, dtype=tf.float32) - w/2, 
            indexing='ij'
        )
        
        # Scale coordinates appropriately
        # Convert to units of lambda/(2*NA) for proper scaling
        scale_factor = self.wavelength / (2 * self.na) / self.n_immersion
        x_scaled = x * scale_factor
        y_scaled = y * scale_factor
        
        # Calculate radial distance
        rho = tf.sqrt(x_scaled**2 + y_scaled**2)
        
        # Calculate the angle
        theta = tf.atan2(y_scaled, x_scaled)
        
        # Calculate the PSF components based on polarization
        if polarization in ['te', 'tm', 'unpolarized']:
            # For simplicity, we'll use scalar approximation with polarization factors
            # In a full vectorial model, we would solve the full Maxwell equations
            
            # Calculate the scalar PSF using Airy function approximation
            airy_factor = self.k0 * self.na * rho
            # Avoid division by zero
            airy_factor = tf.where(airy_factor == 0, 1e-10, airy_factor)
            
            # Scalar PSF based on Airy function
            scalar_psf = tf.pow(2 * tf.bessel_j1(airy_factor) / airy_factor, 2)
            
            # Apply polarization-dependent factors
            if polarization == 'te':
                # TE polarization has uniform response
                return scalar_psf
            elif polarization == 'tm':
                # TM polarization has enhanced axial response
                # Simplified model: slightly enhanced off-axis intensity
                tm_enhancement = 1.0 + 0.3 * tf.pow(tf.sin(tf.asin(rho * self.na / self.na)), 2)
                return scalar_psf * tm_enhancement
            else:  # unpolarized
                # Average of TE and TM responses
                tm_enhancement = 1.0 + 0.3 * tf.pow(tf.sin(tf.asin(tf.clip_by_value(rho * self.na / self.na, 0.0, 1.0))), 2)
                avg_psf = (scalar_psf + scalar_psf * tm_enhancement) / 2
                return avg_psf
        else:
            # For radial/azimuthal polarization, the model would be more complex
            # Returning scalar approximation for now
            airy_factor = self.k0 * self.na * rho
            airy_factor = tf.where(airy_factor == 0, 1e-10, airy_factor)
            scalar_psf = tf.pow(2 * tf.bessel_j1(airy_factor) / airy_factor, 2)
            return scalar_psf


class OpticalSystem:
    """Simplified optical system class for data generation purposes."""
    
    def __init__(
        self,
        wavelength: float = 193e-9,  # DUV wavelength in meters
        na: float = 1.35,  # Numerical aperture
        resolution: float = 8e-9  # Resolution in meters
    ):
        """
        Initialize the optical system.
        
        Args:
            wavelength: Exposure wavelength in meters (193nm)
            na: Numerical aperture of the projection lens
            resolution: Resolution of the simulation in meters per pixel
        """
        self.wavelength = wavelength
        self.na = na
        self.resolution = resolution
        
    def calculate_coherent_transfer_function(self, shape):
        """Calculate the coherent transfer function of the optical system."""
        h, w = shape
        
        # Create coordinate grids
        y, x = tf.meshgrid(
            tf.range(h, dtype=tf.float32), 
            tf.range(w, dtype=tf.float32), 
            indexing='ij'
        )
        
        # Center the coordinates
        center_y, center_x = h / 2.0, w / 2.0
        y = y - center_y
        x = x - center_x
        
        # Convert to spatial frequency units
        # Scale coordinates according to pixel resolution and wavelength
        scale_factor = self.resolution / (self.wavelength / 1.44)  # Scale relative to lambda in immersion
        fx = x * scale_factor
        fy = y * scale_factor
        
        # Calculate radial frequency
        rho = tf.sqrt(fx**2 + fy**2)
        
        # Create circular pupil (frequency cutoff based on NA)
        cutoff_freq = self.na / (self.wavelength / 1.44)  # NA divided by lambda in immersion
        # Normalize by the maximum frequency that can be represented
        max_freq = 0.5 / self.resolution  # Nyquist frequency
        normalized_cutoff = cutoff_freq / max_freq
        
        ctf = tf.cast(rho <= normalized_cutoff, tf.complex64)
        
        return ctf


def calculate_tcc_matrix(
    illumination: IlluminationSource,
    projection_lens: ProjectionLens,
    sampling_factor: int = 2
) -> tf.Tensor:
    """
    Calculate the Transmission Cross Coefficient (TCC) matrix for partially coherent imaging.
    
    Args:
        illumination: Illumination source
        projection_lens: Projection lens model
        sampling_factor: Sampling factor for higher resolution calculation
        
    Returns:
        TCC matrix as a 4D tensor
    """
    # Get the source map and pupil function
    source_map = illumination.get_source_map()
    pupil_function = projection_lens.pupil_function
    
    # Expand dimensions for TCC calculation
    h, w = source_map.shape
    
    # The full TCC calculation is quite complex, involving integrals over the source
    # For this implementation, we'll use a simplified approach
    
    # Zero-pad the source map and pupil function for higher resolution
    padded_h, padded_w = h * sampling_factor, w * sampling_factor
    pad_h, pad_w = (padded_h - h) // 2, (padded_w - w) // 2
    
    padded_source = tf.pad(source_map, [[pad_h, pad_h], [pad_w, pad_w]], mode='CONSTANT')
    padded_pupil = tf.pad(pupil_function, [[pad_h, pad_h], [pad_w, pad_w]], mode='CONSTANT')
    
    # The TCC is calculated as the correlation of pupil functions weighted by the source
    # This is a simplified version - a full implementation would involve more complex integrals
    tcc_matrix = tf.zeros((h, w, h, w), dtype=tf.complex64)
    
    # For computational efficiency, we'll use the Abbe formulation of partial coherence
    # where the effective source is decomposed into discrete points
    for i in range(min(5, h)):  # Limit for efficiency
        for j in range(min(5, w)):  # Limit for efficiency
            # Sample a point in the source
            source_val = source_map[i * h // 5, j * w // 5]
            if source_val > 1e-6:  # Only process significant source points
                # Calculate the contribution of this source point to the TCC
                # This is a simplification of the actual TCC calculation
                shifted_pupil = tf.roll(padded_pupil, shift=[i * sampling_factor, j * sampling_factor], axis=[0, 1])
                
                # Calculate the effective transfer function for this source point
                # TCC contribution = Pupil(u-v) * Pupil_conj(u) * Source_point
                # For simplicity, we'll use a discrete approximation
                pass  # Actual implementation would be quite complex
    
    # Return a simplified diagonal approximation of the TCC for now
    # A full implementation would compute the complete 4D TCC matrix
    diag_approx = tf.eye(h * w, dtype=tf.complex64)
    return tf.reshape(diag_approx, (h, w, h, w))


class HopkinsImagingModel:
    """Implementation of the Hopkins partially coherent imaging model."""
    
    def __init__(
        self,
        illumination: IlluminationSource,
        projection_lens: ProjectionLens
    ):
        """
        Initialize the Hopkins imaging model.
        
        Args:
            illumination: Illumination source
            projection_lens: Projection lens model
        """
        self.illumination = illumination
        self.projection_lens = projection_lens
        self.tcc_matrix = calculate_tcc_matrix(illumination, projection_lens)
        
    def simulate_partially_coherent_image(
        self,
        mask_transmission: tf.Tensor,
        method: str = 'abbe'
    ) -> tf.Tensor:
        """
        Simulate the partially coherent image using the Hopkins model.
        
        Args:
            mask_transmission: Complex mask transmission function
            method: Method to use ('abbe' for Abbe approximation, 'tcc' for full TCC)
            
        Returns:
            Simulated aerial image intensity
        """
        if method == 'abbe':
            # Use Abbe's approximation for partially coherent imaging
            # Image intensity = sum_over_source_points[|FT[Pupil * Shifted_Mask]|^2 * Source_Intensity]
            
            # Get the source map
            source_map = self.illumination.get_source_map()
            h, w = source_map.shape
            
            # For each significant point in the source, calculate the coherent image
            aerial_image = tf.zeros((h, w), dtype=tf.float32)
            
            # Sample significant source points (for efficiency)
            significant_coords = tf.where(source_map > tf.reduce_max(source_map) * 0.1)
            
            # Limit the number of sample points for efficiency
            sample_indices = tf.random.shuffle(tf.range(tf.shape(significant_coords)[0]))
            sample_indices = sample_indices[:min(9, tf.shape(significant_coords)[0])]
            sampled_coords = tf.gather(significant_coords, sample_indices)
            
            for coord in sampled_coords:
                y_idx, x_idx = coord[0], coord[1]
                source_strength = source_map[y_idx, x_idx]
                
                # Calculate the shift corresponding to this source point
                # This is a simplified approach
                shift_y = (tf.cast(y_idx, tf.float32) - h/2) / (h/2)
                shift_x = (tf.cast(x_idx, tf.float32) - w/2) / (w/2)
                
                # Apply shift to the mask (this is a simplification)
                # In reality, this would involve shifting the mask spectrum
                shifted_mask = mask_transmission  # Placeholder
                
                # Calculate the pupil-weighted image
                pupil = self.projection_lens.pupil_function
                pupil_mask_product = pupil * shifted_mask
                
                # Calculate coherent image for this source point
                coherent_image = tf.abs(tf.signal.fft2d(pupil_mask_product)) ** 2
                
                # Add to total aerial image weighted by source strength
                aerial_image += source_strength * coherent_image
            
            # Normalize by total source strength
            total_strength = tf.reduce_sum(source_map)
            aerial_image = aerial_image / total_strength if total_strength > 0 else aerial_image
            
            return aerial_image
        else:  # 'tcc' method would use the full TCC matrix
            # This would involve a more complex computation with the 4D TCC matrix
            # For now, we'll fall back to the Abbe method
            return self.simulate_partially_coherent_image(mask_transmission, method='abbe')