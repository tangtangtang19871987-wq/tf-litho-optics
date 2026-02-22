"""Inverse Lithography Technology (ILT) module for DUV 193nm lithography simulator.

This module implements ILT algorithms for synthesizing mask patterns that produce
desired wafer patterns in immersion lithography systems targeting the 7nm node.
"""

import tensorflow as tf
import numpy as np
from typing import Tuple, Optional, Dict
from lithography_simulator.core import LithographySystem
from lithography_simulator.mask import Mask, BinaryMask
from lithography_simulator.illumination import IlluminationSource
from lithography_simulator.mbopc import MBOPC


class ILT:
    """Inverse Lithography Technology class."""
    
    def __init__(
        self,
        lithography_system: LithographySystem,
        illumination: IlluminationSource,
        target_pattern: tf.Tensor,
        initial_mask: Optional[Mask] = None,
        learning_rate: float = 0.01,
        max_iterations: int = 200,
        regularization_weights: Dict[str, float] = None,
        binary_contrast: float = 10.0,
        constraint_penalty: float = 1.0
    ):
        """
        Initialize the ILT optimizer.
        
        Args:
            lithography_system: The lithography system to simulate
            illumination: Illumination source
            target_pattern: Desired target pattern
            initial_mask: Initial mask guess (if None, will be derived from target)
            learning_rate: Learning rate for gradient descent
            max_iterations: Maximum number of iterations
            regularization_weights: Weights for different regularization terms
            binary_contrast: Contrast parameter for binary enforcement
            constraint_penalty: Penalty weight for constraint violations
        """
        self.lithography_system = lithography_system
        self.illumination = illumination
        self.target_pattern = target_pattern
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.binary_contrast = binary_contrast
        self.constraint_penalty = constraint_penalty
        
        # Set default regularization weights if not provided
        if regularization_weights is None:
            regularization_weights = {
                'edge_smoothness': 0.01,
                'binary_enforcement': 0.1,
                'mask_area': 0.001
            }
        self.reg_weights = regularization_weights
        
        # Initialize mask
        if initial_mask is None:
            # Create an initial mask based on the target pattern
            initial_mask = BinaryMask(target_pattern.shape)
            # Use a simple threshold of the target pattern
            # For complex tensors, compare the magnitude
            binary_target_real = tf.cast(tf.greater(tf.abs(target_pattern), 0.5), tf.float32)
            binary_target = tf.cast(binary_target_real, tf.complex64)
            initial_mask.set_pattern(binary_target)
        
        self.mask = initial_mask
        self.optimization_history = []
        
    def cost_function(
        self,
        predicted_image: tf.Tensor,
        target_image: tf.Tensor,
        mask_pattern: tf.Tensor
    ) -> tf.Tensor:
        """
        Calculate the ILT cost function with multiple regularization terms.
        
        Args:
            predicted_image: Predicted aerial image from simulation
            target_image: Target pattern
            mask_pattern: Current mask pattern
            
        Returns:
            Scalar cost value
        """
        # Image fidelity term (mean squared error)
        image_fidelity = tf.reduce_mean(tf.square(predicted_image - target_image))
        
        # Extract real part of mask for regularization calculations
        mask_real = tf.math.real(mask_pattern)
        
        # 1. Edge smoothness regularization (total variation)
        diff_x = mask_real[:, 1:] - mask_real[:, :-1]
        diff_y = mask_real[1:, :] - mask_real[:-1, :]
        edge_smoothness = tf.reduce_mean(tf.abs(diff_x)) + tf.reduce_mean(tf.abs(diff_y))
        
        # 2. Binary enforcement (push values toward 0 or 1)
        # Use sigmoid to create a smooth approximation of binary enforcement
        binary_loss = tf.reduce_mean(tf.square(mask_real * (1.0 - mask_real)))
        
        # 3. Mask area regularization (prevent overly large or small features)
        mask_area_deviation = tf.square(tf.reduce_mean(mask_real) - 0.5)  # Encourage 50% fill ratio
        
        # Combine all terms
        total_cost = (
            image_fidelity +
            self.reg_weights['edge_smoothness'] * edge_smoothness +
            self.reg_weights['binary_enforcement'] * binary_loss +
            self.reg_weights['mask_area'] * mask_area_deviation
        )
        
        return total_cost
    
    def optimize_step(self) -> Tuple[tf.Tensor, float]:
        """
        Perform one ILT optimization step using gradient descent.
        
        Returns:
            Tuple of (optimized mask pattern, cost value)
        """
        with tf.GradientTape() as tape:
            tape.watch(self.mask.pattern)
            
            # Simulate the current mask
            aerial_image = self.lithography_system.simulate_aerial_image(self.mask.pattern)
            
            # Calculate cost
            cost = self.cost_function(aerial_image, self.target_pattern, self.mask.pattern)
        
        # Calculate gradients
        gradients = tape.gradient(cost, self.mask.pattern)
        
        # Update mask pattern using gradient descent
        new_pattern = self.mask.pattern - self.learning_rate * gradients
        
        # Enforce mask constraints (binary values, magnitude bounds)
        new_pattern = self._enforce_mask_constraints(new_pattern)
        
        # Update the mask
        self.mask.set_pattern(new_pattern)
        
        return new_pattern, cost.numpy()
    
    def _enforce_mask_constraints(self, mask_pattern: tf.Tensor) -> tf.Tensor:
        """
        Enforce physical constraints on the mask pattern.
        
        Args:
            mask_pattern: Raw mask pattern from optimization
            
        Returns:
            Mask pattern with constraints enforced
        """
        # Extract real and imaginary parts
        real_part = tf.math.real(mask_pattern)
        imag_part = tf.math.imag(mask_pattern)
        
        # Enforce binary constraint on real part (transmission)
        # Use sigmoid to smoothly enforce binary values
        real_part = tf.nn.sigmoid(self.binary_contrast * (real_part - 0.5))
        
        # Constrain imaginary part to be small (minimal phase variations for now)
        imag_part = tf.clip_by_value(imag_part, -0.1, 0.1)
        
        # Combine back into complex number
        constrained_pattern = tf.complex(real_part, imag_part)
        
        return constrained_pattern
    
    def run_optimization(self) -> Tuple[Mask, Dict[str, list]]:
        """
        Run the full ILT optimization.
        
        Returns:
            Tuple of (optimized mask, optimization history)
        """
        print("Starting ILT optimization...")
        
        for iteration in range(self.max_iterations):
            _, cost = self.optimize_step()
            
            # Record cost for monitoring convergence
            self.optimization_history.append({
                'iteration': iteration,
                'cost': cost,
                'max_change': self.learning_rate * tf.reduce_max(tf.abs(self.mask.pattern))
            })
            
            if iteration % 20 == 0:
                print(f"Iteration {iteration}, Cost: {cost:.6f}")
                
            # Check for convergence (simple check based on cost change)
            if len(self.optimization_history) > 1:
                prev_cost = self.optimization_history[-2]['cost']
                if abs(prev_cost - cost) < 1e-8:
                    print(f"Converged at iteration {iteration}")
                    break
        
        print("ILT optimization completed.")
        
        # Return the optimized mask and history
        return self.mask, {
            'cost_history': [h['cost'] for h in self.optimization_history],
            'iteration_numbers': [h['iteration'] for h in self.optimization_history],
            'max_changes': [h['max_change'] for h in self.optimization_history]
        }


class PixelBasedILT(ILT):
    """Pixel-based ILT implementation using level-set methods."""
    
    def __init__(
        self,
        lithography_system: LithographySystem,
        illumination: IlluminationSource,
        target_pattern: tf.Tensor,
        initial_mask: Optional[Mask] = None,
        learning_rate: float = 0.01,
        max_iterations: int = 200,
        regularization_weights: Dict[str, float] = None,
        level_set_threshold: float = 0.5
    ):
        """
        Initialize the pixel-based ILT optimizer.
        
        Args:
            lithography_system: The lithography system to simulate
            illumination: Illumination source
            target_pattern: Desired target pattern
            initial_mask: Initial mask guess
            learning_rate: Learning rate for optimization
            max_iterations: Maximum number of iterations
            regularization_weights: Weights for regularization terms
            level_set_threshold: Threshold for level-set based binary enforcement
        """
        super().__init__(
            lithography_system, illumination, target_pattern, initial_mask,
            learning_rate, max_iterations, regularization_weights
        )
        self.level_set_threshold = level_set_threshold
        # Use a level-set function instead of direct pixel optimization
        if initial_mask is None:
            self.level_set_func = tf.Variable(
                tf.random.uniform(target_pattern.shape, minval=-1, maxval=1, dtype=tf.float32),
                trainable=True
            )
        else:
            # Initialize level set from initial mask
            mask_real = tf.math.real(initial_mask.pattern)
            self.level_set_func = tf.Variable(mask_real * 2 - 1, trainable=True)  # Scale to [-1, 1]
    
    def _update_mask_from_level_set(self):
        """Update the mask pattern based on the level-set function."""
        # Convert level-set function to binary mask
        binary_mask = tf.cast(self.level_set_func > self.level_set_threshold, tf.complex64)
        # Add small imaginary component to maintain complex nature
        self.mask.set_pattern(binary_mask + 0.01j * tf.cast(binary_mask, tf.complex64))
    
    def optimize_step(self) -> Tuple[tf.Tensor, float]:
        """
        Perform one optimization step using level-set representation.
        
        Returns:
            Tuple of (optimized mask pattern, cost value)
        """
        with tf.GradientTape() as tape:
            tape.watch(self.level_set_func)
            
            # Update mask from level set
            self._update_mask_from_level_set()
            
            # Simulate the current mask
            aerial_image = self.lithography_system.simulate_aerial_image(self.mask.pattern)
            
            # Calculate cost
            cost = self.cost_function(aerial_image, self.target_pattern, self.mask.pattern)
        
        # Calculate gradients with respect to level-set function
        gradients = tape.gradient(cost, self.level_set_func)
        
        # Update level-set function using gradient descent
        self.level_set_func.assign_sub(self.learning_rate * gradients)
        
        # Update mask based on new level-set function
        self._update_mask_from_level_set()
        
        return self.mask.pattern, cost.numpy()


class SourceMaskOptimization:
    """Joint optimization of source and mask for improved imaging performance."""
    
    def __init__(
        self,
        lithography_system: LithographySystem,
        initial_illumination: IlluminationSource,
        target_pattern: tf.Tensor,
        initial_mask: Optional[Mask] = None,
        learning_rate_source: float = 0.001,
        learning_rate_mask: float = 0.01,
        max_iterations: int = 100
    ):
        """
        Initialize joint source-mask optimization.
        
        Args:
            lithography_system: The lithography system to simulate
            initial_illumination: Initial illumination source
            target_pattern: Desired target pattern
            initial_mask: Initial mask guess
            learning_rate_source: Learning rate for source optimization
            learning_rate_mask: Learning rate for mask optimization
            max_iterations: Maximum number of iterations
        """
        self.lithography_system = lithography_system
        self.initial_illumination = initial_illumination
        self.target_pattern = target_pattern
        self.learning_rate_source = learning_rate_source
        self.learning_rate_mask = learning_rate_mask
        self.max_iterations = max_iterations
        
        # Initialize mask
        if initial_mask is None:
            initial_mask = BinaryMask(target_pattern.shape)
            # For complex tensors, compare the magnitude
            binary_target_real = tf.cast(tf.greater(tf.abs(target_pattern), 0.5), tf.float32)
            binary_target = tf.cast(binary_target_real, tf.complex64)
            initial_mask.set_pattern(binary_target)
        
        self.mask = initial_mask
        self.optimization_history = []
        
        # For this implementation, we'll optimize a simplified source parameter
        # In practice, this would involve optimizing the full source map
        self.source_params = tf.Variable([0.5, 0.3], dtype=tf.float32, trainable=True)  # sigma_in, sigma_out for annular
    
    def _create_updated_illumination(self) -> IlluminationSource:
        """Create an updated illumination source based on current parameters."""
        from lithography_simulator.illumination import PartialCoherentSource
        
        # Create new illumination with current parameters
        # Clamp values to physically meaningful ranges
        sigma_in = tf.clip_by_value(self.source_params[0], 0.1, 0.9)
        sigma_out = tf.clip_by_value(self.source_params[1] + sigma_in, sigma_in + 0.05, 1.0)
        
        return PartialCoherentSource(
            self.target_pattern.shape,
            sigma=sigma_out.numpy(),
            source_type='annular'
        )
    
    def cost_function(
        self,
        predicted_image: tf.Tensor,
        target_image: tf.Tensor
    ) -> tf.Tensor:
        """
        Calculate cost function for joint optimization.
        
        Args:
            predicted_image: Predicted aerial image from simulation
            target_image: Target pattern
            
        Returns:
            Scalar cost value
        """
        # Image fidelity term
        image_fidelity = tf.reduce_mean(tf.square(predicted_image - target_image))
        
        # Additional penalties could be added here for source characteristics
        # e.g., source smoothness, manufacturing constraints
        
        return image_fidelity
    
    def run_joint_optimization(self) -> Tuple[Mask, IlluminationSource, Dict[str, list]]:
        """
        Run joint source-mask optimization.
        
        Returns:
            Tuple of (optimized mask, optimized illumination, optimization history)
        """
        print("Starting Joint Source-Mask Optimization...")
        
        for iteration in range(self.max_iterations):
            with tf.GradientTape(persistent=True) as tape:
                # Create updated illumination
                current_illumination = self._create_updated_illumination()
                
                # Simulate with current mask and illumination
                # For this implementation, we'll use the system's basic simulation
                aerial_image = self.lithography_system.simulate_aerial_image(self.mask.pattern)
                
                # Calculate cost
                cost = self.cost_function(aerial_image, self.target_pattern)
            
            # Calculate gradients for mask and source parameters
            grad_mask = tape.gradient(cost, self.mask.pattern)
            grad_source = tape.gradient(cost, self.source_params)
            
            # Update mask
            new_pattern = self.mask.pattern - self.learning_rate_mask * grad_mask
            # Enforce constraints on mask
            real_part = tf.nn.sigmoid(10.0 * (tf.math.real(new_pattern) - 0.5))
            imag_part = tf.clip_by_value(tf.math.imag(new_pattern), -0.1, 0.1)
            new_pattern = tf.complex(real_part, imag_part)
            self.mask.set_pattern(new_pattern)
            
            # Update source parameters
            self.source_params.assign_sub(self.learning_rate_source * grad_source)
            
            # Delete tape to free memory
            del tape
            
            # Record cost for monitoring convergence
            self.optimization_history.append({
                'iteration': iteration,
                'cost': cost.numpy(),
                'sigma_in': tf.clip_by_value(self.source_params[0], 0.1, 0.9).numpy(),
                'sigma_out': tf.clip_by_value(self.source_params[1] + self.source_params[0], 
                                              tf.clip_by_value(self.source_params[0], 0.1, 0.9) + 0.05, 1.0).numpy()
            })
            
            if iteration % 20 == 0:
                print(f"Iteration {iteration}, Cost: {cost.numpy():.6f}, "
                      f"Sigma in: {self.optimization_history[-1]['sigma_in']:.3f}, "
                      f"Sigma out: {self.optimization_history[-1]['sigma_out']:.3f}")
        
        print("Joint optimization completed.")
        
        # Create final illumination with optimized parameters
        final_illumination = self._create_updated_illumination()
        
        return self.mask, final_illumination, {
            'cost_history': [h['cost'] for h in self.optimization_history],
            'iteration_numbers': [h['iteration'] for h in self.optimization_history],
            'sigma_in_history': [h['sigma_in'] for h in self.optimization_history],
            'sigma_out_history': [h['sigma_out'] for h in self.optimization_history]
        }