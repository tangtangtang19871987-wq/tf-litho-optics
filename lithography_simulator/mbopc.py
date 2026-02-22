"""Model-Based Optical Proximity Correction (MBOPC) module for DUV 193nm lithography simulator.

This module implements gradient-based OPC algorithms for correcting optical proximity effects
in immersion lithography systems targeting the 7nm node.
"""

import tensorflow as tf
import numpy as np
from typing import Tuple, Optional, Callable, Dict
from lithography_simulator.core import LithographySystem
from lithography_simulator.mask import Mask, BinaryMask
from lithography_simulator.illumination import IlluminationSource
from lithography_simulator.optics import HopkinsImagingModel


class MBOPC:
    """Model-Based Optical Proximity Correction class."""
    
    def __init__(
        self,
        lithography_system: LithographySystem,
        illumination: IlluminationSource,
        target_pattern: tf.Tensor,
        initial_mask: Optional[Mask] = None,
        learning_rate: float = 0.01,
        max_iterations: int = 100,
        regularization_weight: float = 0.01,
        process_window_focus_range: Tuple[float, float] = (-0.05e-6, 0.05e-6),  # ±50nm
        process_window_dose_range: Tuple[float, float] = (0.9, 1.1)  # ±10%
    ):
        """
        Initialize the MBOPC optimizer.
        
        Args:
            lithography_system: The lithography system to simulate
            illumination: Illumination source
            target_pattern: Desired target pattern
            initial_mask: Initial mask guess (if None, will be derived from target)
            learning_rate: Learning rate for gradient descent
            max_iterations: Maximum number of iterations
            regularization_weight: Weight for regularization terms
            process_window_focus_range: Range of focus values to consider
            process_window_dose_range: Range of dose values to consider
        """
        self.lithography_system = lithography_system
        self.illumination = illumination
        self.target_pattern = target_pattern
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.regularization_weight = regularization_weight
        self.process_window_focus_range = process_window_focus_range
        self.process_window_dose_range = process_window_dose_range
        
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
        mask_pattern: tf.Tensor,
        regularization_weight: float = 0.01
    ) -> tf.Tensor:
        """
        Calculate the cost function combining image fidelity and mask regularization.
        
        Args:
            predicted_image: Predicted aerial image from simulation
            target_image: Target pattern
            mask_pattern: Current mask pattern
            regularization_weight: Weight for regularization terms
            
        Returns:
            Scalar cost value
        """
        # Image fidelity term (mean squared error)
        image_fidelity = tf.reduce_mean(tf.square(predicted_image - target_image))
        
        # Regularization terms to ensure mask manufacturability
        # Extract real part of mask for regularization calculations
        mask_real = tf.math.real(mask_pattern)
        
        # 1. Mask complexity regularization (penalize rapid changes)
        mask_diff_x = mask_real[:, 1:] - mask_real[:, :-1]
        mask_diff_y = mask_real[1:, :] - mask_real[:-1, :]
        mask_complexity = tf.reduce_mean(tf.square(mask_diff_x)) + tf.reduce_mean(tf.square(mask_diff_y))
        
        # 2. Mask value regularization (encourage binary values)
        mask_binary_penalty = tf.reduce_mean(tf.square(mask_real * (1 - mask_real)))
        
        # Total cost
        total_cost = image_fidelity + regularization_weight * (mask_complexity + mask_binary_penalty)
        
        return total_cost
    
    def optimize_step(self) -> Tuple[tf.Tensor, float]:
        """
        Perform one optimization step using gradient descent.
        
        Returns:
            Tuple of (optimized mask pattern, cost value)
        """
        with tf.GradientTape() as tape:
            tape.watch(self.mask.pattern)
            
            # Simulate the current mask
            aerial_image = self.lithography_system.simulate_aerial_image(self.mask.pattern)
            
            # Calculate cost
            cost = self.cost_function(
                aerial_image, 
                self.target_pattern, 
                self.mask.pattern, 
                self.regularization_weight
            )
        
        # Calculate gradients
        gradients = tape.gradient(cost, self.mask.pattern)
        
        # Update mask pattern using gradient descent
        new_pattern = self.mask.pattern - self.learning_rate * gradients
        
        # Project onto feasible set (enforce binary-like constraints)
        # For complex masks, we might want to constrain magnitude and phase separately
        magnitude = tf.abs(new_pattern)
        phase = tf.math.angle(new_pattern)
        
        # Constrain magnitude to be between 0 and 1
        magnitude = tf.clip_by_value(magnitude, 0.0, 1.0)
        
        # Reconstruct complex pattern
        # Use tf.cos and tf.sin to build the complex exponential
        real_part = magnitude * tf.cos(phase)
        imag_part = magnitude * tf.sin(phase)
        new_pattern = tf.complex(real_part, imag_part)
        
        # Update the mask
        self.mask.set_pattern(new_pattern)
        
        return new_pattern, cost.numpy()
    
    def run_optimization(self) -> Tuple[Mask, Dict[str, list]]:
        """
        Run the full OPC optimization.
        
        Returns:
            Tuple of (optimized mask, optimization history)
        """
        print("Starting MBOPC optimization...")
        
        for iteration in range(self.max_iterations):
            _, cost = self.optimize_step()
            
            # Record cost for monitoring convergence
            self.optimization_history.append({
                'iteration': iteration,
                'cost': cost,
                'max_change': self.learning_rate * tf.reduce_max(tf.abs(self.mask.pattern))
            })
            
            if iteration % 10 == 0:
                print(f"Iteration {iteration}, Cost: {cost:.6f}")
                
            # Check for convergence (simple check based on cost change)
            if len(self.optimization_history) > 1:
                prev_cost = self.optimization_history[-2]['cost']
                if abs(prev_cost - cost) < 1e-8:
                    print(f"Converged at iteration {iteration}")
                    break
        
        print("Optimization completed.")
        
        # Return the optimized mask and history
        return self.mask, {
            'cost_history': [h['cost'] for h in self.optimization_history],
            'iteration_numbers': [h['iteration'] for h in self.optimization_history],
            'max_changes': [h['max_change'] for h in self.optimization_history]
        }


class EdgePlacementErrorCalculator:
    """Calculator for Edge Placement Error (EPE) metrics."""
    
    def __init__(self, pixel_size: float = 1e-9):
        """
        Initialize the EPE calculator.
        
        Args:
            pixel_size: Physical size of each pixel in meters
        """
        self.pixel_size = pixel_size
    
    def calculate_epe(
        self, 
        predicted_edges: tf.Tensor, 
        target_edges: tf.Tensor
    ) -> Dict[str, float]:
        """
        Calculate Edge Placement Error between predicted and target edges.
        
        Args:
            predicted_edges: Predicted edge positions (binary mask)
            target_edges: Target edge positions (binary mask)
            
        Returns:
            Dictionary with EPE metrics
        """
        # Convert to numpy for distance transform calculations
        pred_np = tf.cast(predicted_edges > 0.5, tf.float32).numpy()
        target_np = tf.cast(target_edges > 0.5, tf.float32).numpy()
        
        # Calculate signed distance transforms
        from scipy.ndimage import distance_transform_edt
        
        # Positive distances: distance from target edge to predicted edge
        pos_distances = distance_transform_edt(1 - target_np) * self.pixel_size
        pos_distances = pos_distances * pred_np  # Only measure where predicted edge is
        
        # Negative distances: distance from predicted edge to target edge  
        neg_distances = distance_transform_edt(1 - pred_np) * self.pixel_size
        neg_distances = neg_distances * target_np  # Only measure where target edge is
        
        # Combine distances (positive when predicted edge is outside target, negative when inside)
        combined_distances = pos_distances - neg_distances
        
        # Calculate statistics
        epe_rms = np.sqrt(np.mean(combined_distances ** 2))
        epe_mean = np.mean(combined_distances)
        epe_max = np.max(np.abs(combined_distances))
        epe_3sigma = 3 * np.std(combined_distances)
        
        return {
            'epe_rms': epe_rms,
            'epe_mean': epe_mean,
            'epe_max': epe_max,
            'epe_3sigma': epe_3sigma,
            'pixel_size': self.pixel_size
        }


class ProcessWindowOptimizer:
    """Optimizer that considers the process window during OPC."""
    
    def __init__(
        self,
        mbopc_optimizer: MBOPC,
        focus_steps: int = 5,
        dose_steps: int = 5
    ):
        """
        Initialize the process window optimizer.
        
        Args:
            mbopc_optimizer: Base MBOPC optimizer
            focus_steps: Number of focus steps to consider
            dose_steps: Number of dose steps to consider
        """
        self.mbopc_optimizer = mbopc_optimizer
        self.focus_steps = focus_steps
        self.dose_steps = dose_steps
    
    def calculate_process_window_cost(
        self,
        mask_pattern: tf.Tensor,
        target_pattern: tf.Tensor
    ) -> tf.Tensor:
        """
        Calculate cost considering variations in focus and dose.
        
        Args:
            mask_pattern: Current mask pattern
            target_pattern: Target pattern
            
        Returns:
            Cost considering process window variations
        """
        costs = []
        
        # Generate focus and dose variations
        focus_range = tf.linspace(
            self.mbopc_optimizer.process_window_focus_range[0],
            self.mbopc_optimizer.process_window_focus_range[1],
            self.focus_steps
        )
        
        dose_range = tf.linspace(
            self.mbopc_optimizer.process_window_dose_range[0],
            self.mbopc_optimizer.process_window_dose_range[1],
            self.dose_steps
        )
        
        # Evaluate cost at each process condition
        for focus in focus_range:
            for dose in dose_range:
                # Modify the lithography system for this condition
                # For now, we'll just simulate at different focus values
                aerial_image = self.mbopc_optimizer.lithography_system.simulate_aerial_image(
                    mask_pattern, focus_offset=focus.numpy()
                )
                
                # Scale for dose variation (simplified model)
                aerial_image = aerial_image * dose
                
                # Calculate fidelity cost
                # Take the absolute value of target pattern to match the real-valued aerial image
                fidelity_cost = tf.reduce_mean(tf.square(aerial_image - tf.abs(target_pattern)))
                costs.append(fidelity_cost)
        
        # Return mean cost across process window
        mean_cost = tf.reduce_mean(costs)
        
        # Also add variance penalty to encourage robustness
        variance_penalty = tf.math.reduce_std(costs) * 0.1  # Weight for robustness
        
        return mean_cost + variance_penalty
    
    def run_process_window_optimization(self) -> Tuple[Mask, Dict[str, list]]:
        """
        Run OPC optimization considering the process window.
        
        Returns:
            Tuple of (optimized mask, optimization history)
        """
        print("Starting Process Window OPC optimization...")
        
        optimization_history = []
        
        # Temporarily replace the cost function in the optimizer
        original_cost_fn = self.mbopc_optimizer.cost_function
        
        def pw_cost_function(pred_img, target_img, mask_pat, reg_weight):
            # Use the process window cost function
            return self.calculate_process_window_cost(mask_pat, target_img)
        
        self.mbopc_optimizer.cost_function = pw_cost_function
        
        # Run optimization
        optimized_mask, history = self.mbopc_optimizer.run_optimization()
        
        # Restore original cost function
        self.mbopc_optimizer.cost_function = original_cost_fn
        
        return optimized_mask, history