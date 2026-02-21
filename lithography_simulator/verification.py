"""Verification and Metrics module for DUV 193nm lithography simulator.

This module implements verification metrics and analysis tools for evaluating
the performance of OPC and ILT corrections in immersion lithography systems
targeting the 7nm node.
"""

import tensorflow as tf
import numpy as np
from typing import Tuple, Dict, Optional
from scipy import ndimage
from scipy.ndimage import distance_transform_edt
from skimage import measure, morphology
from lithography_simulator.core import LithographySystem
from lithography_simulator.mask import Mask


def calculate_critical_dimensions(
    pattern: np.ndarray, 
    pixel_size: float = 1e-9,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculate critical dimensions of a pattern.
    
    Args:
        pattern: 2D array representing the pattern
        pixel_size: Physical size of each pixel in meters
        threshold: Threshold for binarization
        
    Returns:
        Dictionary containing CD measurements
    """
    # Binarize the pattern
    binary_pattern = (pattern > threshold).astype(int)
    
    # Label connected components
    labeled_pattern, num_features = ndimage.label(binary_pattern)
    
    # Calculate CD for each feature
    cds = []
    pitches = []
    
    # Find horizontal lines (for line/space patterns)
    for i in range(num_features):
        feature_idx = i + 1
        feature_coords = np.where(labeled_pattern == feature_idx)
        
        if len(feature_coords[0]) > 0:
            min_row, max_row = np.min(feature_coords[0]), np.max(feature_coords[0])
            min_col, max_col = np.min(feature_coords[1]), np.max(feature_coords[1])
            
            # Calculate width and height of bounding box
            width = (max_col - min_col + 1) * pixel_size
            height = (max_row - min_row + 1) * pixel_size
            
            # For line/space patterns, we typically measure the narrowest dimension
            cds.append(min(width, height))
    
    if len(cds) > 0:
        return {
            'cd_min': min(cds),
            'cd_max': max(cds),
            'cd_mean': np.mean(cds),
            'cd_std': np.std(cds),
            'num_features': len(cds)
        }
    else:
        return {
            'cd_min': 0.0,
            'cd_max': 0.0,
            'cd_mean': 0.0,
            'cd_std': 0.0,
            'num_features': 0
        }


def calculate_line_edge_roughness(
    pattern: np.ndarray, 
    threshold: float = 0.5,
    pixel_size: float = 1e-9
) -> Dict[str, float]:
    """
    Calculate Line Edge Roughness (LER) for a pattern.
    
    Args:
        pattern: 2D array representing the pattern
        threshold: Threshold for binarization
        pixel_size: Physical size of each pixel in meters
        
    Returns:
        Dictionary containing LER metrics
    """
    # Binarize the pattern
    binary_pattern = (pattern > threshold).astype(int)
    
    # Find edges using morphological operations
    selem = morphology.disk(1)
    edges = binary_pattern - ndimage.binary_erosion(binary_pattern, structure=selem)
    
    # Label edge pixels
    labeled_edges, num_edge_features = ndimage.label(edges)
    
    # Calculate LER by measuring deviations from ideal straight lines
    # For this implementation, we'll use a simplified approach
    # finding the distance from edge pixels to the nearest interior point
    
    # Calculate distance transform of the binary pattern
    dist_to_exterior = distance_transform_edt(binary_pattern)
    dist_to_interior = distance_transform_edt(1 - binary_pattern)
    
    # Combine to get distance to nearest boundary
    combined_dist = dist_to_interior + dist_to_exterior
    
    # Get edge pixels
    edge_pixels = np.where(edges > 0)
    if len(edge_pixels[0]) == 0:
        return {
            'ler_mean': 0.0,
            'ler_std': 0.0,
            'ler_rms': 0.0,
            'num_edge_pixels': 0
        }
    
    # Calculate LER as deviation from ideal edge position
    edge_distances = combined_dist[edge_pixels]
    
    return {
        'ler_mean': np.mean(edge_distances) * pixel_size,
        'ler_std': np.std(edge_distances) * pixel_size,
        'ler_rms': np.sqrt(np.mean(edge_distances**2)) * pixel_size,
        'num_edge_pixels': len(edge_distances)
    }


def calculate_overlap_area(
    predicted_pattern: np.ndarray,
    target_pattern: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculate overlap metrics between predicted and target patterns.
    
    Args:
        predicted_pattern: 2D array representing the predicted pattern
        target_pattern: 2D array representing the target pattern
        threshold: Threshold for binarization
        
    Returns:
        Dictionary containing overlap metrics
    """
    # Binarize both patterns
    pred_binary = (predicted_pattern > threshold).astype(int)
    target_binary = (target_pattern > threshold).astype(int)
    
    # Calculate intersection and union
    intersection = np.logical_and(pred_binary, target_binary)
    union = np.logical_or(pred_binary, target_binary)
    
    # Calculate metrics
    intersection_area = np.sum(intersection)
    union_area = np.sum(union)
    target_area = np.sum(target_binary)
    pred_area = np.sum(pred_binary)
    
    # Calculate various metrics
    jaccard_index = intersection_area / union_area if union_area > 0 else 0.0
    dice_coefficient = (2 * intersection_area) / (pred_area + target_area) if (pred_area + target_area) > 0 else 0.0
    
    return {
        'jaccard_index': jaccard_index,
        'dice_coefficient': dice_coefficient,
        'intersection_area': intersection_area,
        'union_area': union_area,
        'overlap_ratio': intersection_area / target_area if target_area > 0 else 0.0,
        'target_area': target_area,
        'predicted_area': pred_area
    }


def calculate_process_window(
    lithography_system: LithographySystem,
    mask: Mask,
    target_pattern: np.ndarray,
    focus_range: Tuple[float, float] = (-100e-9, 100e-9),  # ±100nm
    dose_range: Tuple[float, float] = (0.8, 1.2),  # ±20%
    focus_steps: int = 11,
    dose_steps: int = 11
) -> Dict[str, float]:
    """
    Calculate the process window for a given mask.
    
    Args:
        lithography_system: Lithography system to simulate
        mask: Mask to evaluate
        target_pattern: Target pattern for comparison
        focus_range: Range of focus values to test (min, max) in meters
        dose_range: Range of dose values to test (min, max) as multiplier
        focus_steps: Number of focus steps to evaluate
        dose_steps: Number of dose steps to evaluate
        
    Returns:
        Dictionary containing process window metrics
    """
    focus_values = np.linspace(focus_range[0], focus_range[1], focus_steps)
    dose_values = np.linspace(dose_range[0], dose_range[1], dose_steps)
    
    # Calculate success for each condition
    successes = 0
    total_evaluations = 0
    
    # Track worst-case metrics
    worst_overlap = float('inf')
    worst_cd_error = float('inf')
    
    for focus in focus_values:
        for dose in dose_values:
            total_evaluations += 1
            
            # Simulate with current focus and dose
            aerial_image = lithography_system.simulate_aerial_image(
                mask.pattern, focus_offset=focus
            )
            
            # Apply dose scaling
            aerial_image = aerial_image * dose
            
            # Calculate metrics
            overlap_metrics = calculate_overlap_area(aerial_image.numpy(), target_pattern)
            cd_metrics = calculate_critical_dimensions(aerial_image.numpy())
            target_cd_metrics = calculate_critical_dimensions(target_pattern)
            
            # Determine if this condition passes (arbitrary thresholds)
            overlap_pass = overlap_metrics['jaccard_index'] > 0.8
            cd_pass = abs(cd_metrics['cd_mean'] - target_cd_metrics['cd_mean']) < 0.1e-9 if target_cd_metrics['cd_mean'] > 0 else True
            
            if overlap_pass and cd_pass:
                successes += 1
            
            # Update worst-case metrics
            current_overlap = 1 - overlap_metrics['jaccard_index']
            current_cd_error = abs(cd_metrics['cd_mean'] - target_cd_metrics['cd_mean']) if target_cd_metrics['cd_mean'] > 0 else float('inf')
            
            if current_overlap < worst_overlap:
                worst_overlap = current_overlap
            if current_cd_error < worst_cd_error and not np.isinf(current_cd_error):
                worst_cd_error = current_cd_error
    
    process_window = (successes / total_evaluations) * 100 if total_evaluations > 0 else 0.0
    
    return {
        'process_window_percent': process_window,
        'success_count': successes,
        'total_evaluations': total_evaluations,
        'focus_depth': focus_range[1] - focus_range[0],
        'dose_window': dose_range[1] - dose_range[0],
        'worst_case_overlap_error': worst_overlap,
        'worst_case_cd_error': worst_cd_error
    }


def calculate_mask_error_factor(
    aerial_image: np.ndarray,
    target_pattern: np.ndarray,
    mask_pattern: np.ndarray
) -> Dict[str, float]:
    """
    Calculate Mask Error Factor (MEF) metrics.
    
    Args:
        aerial_image: Simulated aerial image
        target_pattern: Target pattern
        mask_pattern: Original mask pattern
        
    Returns:
        Dictionary containing MEF metrics
    """
    # Calculate pattern fidelity
    overlap_metrics = calculate_overlap_area(aerial_image, target_pattern)
    
    # Calculate edge placement error
    # This is a simplified version - a full implementation would be more complex
    target_edges = measure.find_contours(target_pattern, level=0.5)
    image_edges = measure.find_contours(aerial_image, level=0.5)
    
    # As a proxy for MEF, calculate how much variation there is in the image
    # compared to the mask
    image_variation = np.std(aerial_image)
    mask_variation = np.std(np.abs(mask_pattern))
    
    mef = image_variation / mask_variation if mask_variation > 0 else 0.0
    
    return {
        'mef': mef,
        'pattern_fidelity': overlap_metrics['jaccard_index'],
        'image_variation': image_variation,
        'mask_variation': mask_variation
    }


def calculate_resolution_limits(
    lithography_system: LithographySystem,
    pixel_size: float = 1e-9
) -> Dict[str, float]:
    """
    Calculate theoretical resolution limits of the system.
    
    Args:
        lithography_system: Lithography system to evaluate
        pixel_size: Physical size of simulation pixels
        
    Returns:
        Dictionary containing resolution metrics
    """
    # Calculate Rayleigh resolution limit
    # Resolution = k1 * lambda / NA
    # For k1 = 0.25 (typical for advanced processes)
    k1 = 0.25
    resolution_limit = k1 * lithography_system.effective_wavelength / lithography_system.na
    
    # Calculate depth of focus
    # DOF = +/- lambda / (2 * NA^2)
    dof = lithography_system.effective_wavelength / (2 * lithography_system.na**2)
    
    return {
        'rayleigh_resolution_limit': resolution_limit,
        'depth_of_focus': dof,
        'numerical_aperture': lithography_system.na,
        'wavelength': lithography_system.wavelength,
        'effective_wavelength': lithography_system.effective_wavelength,
        'k1_factor': k1
    }


def comprehensive_verification_report(
    lithography_system: LithographySystem,
    mask: Mask,
    target_pattern: np.ndarray,
    simulated_image: np.ndarray,
    pixel_size: float = 1e-9
) -> Dict[str, Dict[str, float]]:
    """
    Generate a comprehensive verification report.
    
    Args:
        lithography_system: Lithography system used for simulation
        mask: Mask that was optimized
        target_pattern: Target pattern
        simulated_image: Simulated aerial image
        pixel_size: Physical size of simulation pixels
        
    Returns:
        Dictionary containing all verification metrics
    """
    # Calculate all metrics
    cd_metrics = calculate_critical_dimensions(simulated_image, pixel_size)
    ler_metrics = calculate_line_edge_roughness(simulated_image, pixel_size=pixel_size)
    overlap_metrics = calculate_overlap_area(simulated_image, target_pattern)
    process_window_metrics = calculate_process_window(lithography_system, mask, target_pattern)
    mef_metrics = calculate_mask_error_factor(simulated_image, target_pattern, mask.pattern.numpy())
    resolution_metrics = calculate_resolution_limits(lithography_system, pixel_size)
    
    # Compile comprehensive report
    report = {
        'critical_dimensions': cd_metrics,
        'line_edge_roughness': ler_metrics,
        'pattern_fidelity': overlap_metrics,
        'process_window': process_window_metrics,
        'mask_error_factor': mef_metrics,
        'resolution_limits': resolution_metrics
    }
    
    return report


class VerificationMetrics:
    """Class for organizing and managing verification metrics."""
    
    def __init__(self, pixel_size: float = 1e-9):
        """
        Initialize verification metrics calculator.
        
        Args:
            pixel_size: Physical size of simulation pixels in meters
        """
        self.pixel_size = pixel_size
    
    def evaluate_mask_performance(
        self,
        lithography_system: LithographySystem,
        mask: Mask,
        target_pattern: np.ndarray,
        simulated_image: Optional[np.ndarray] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate the overall performance of a mask.
        
        Args:
            lithography_system: Lithography system used for simulation
            mask: Mask to evaluate
            target_pattern: Target pattern
            simulated_image: Pre-computed simulated image (optional)
            
        Returns:
            Dictionary of evaluation metrics
        """
        if simulated_image is None:
            # Simulate the image if not provided
            aerial_image_tensor = lithography_system.simulate_aerial_image(mask.pattern)
            simulated_image = aerial_image_tensor.numpy()
        
        # Generate comprehensive report
        report = comprehensive_verification_report(
            lithography_system, mask, target_pattern, simulated_image, self.pixel_size
        )
        
        return report
    
    def compare_masks(
        self,
        lithography_system: LithographySystem,
        mask1: Mask,
        mask2: Mask,
        target_pattern: np.ndarray
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Compare the performance of two masks.
        
        Args:
            lithography_system: Lithography system used for simulation
            mask1: First mask to compare
            mask2: Second mask to compare
            target_pattern: Target pattern
            
        Returns:
            Dictionary comparing metrics for both masks
        """
        # Evaluate both masks
        eval1 = self.evaluate_mask_performance(lithography_system, mask1, target_pattern)
        eval2 = self.evaluate_mask_performance(lithography_system, mask2, target_pattern)
        
        # Calculate differences
        comparison = {}
        for category in eval1:
            comparison[category] = {
                'mask1': eval1[category],
                'mask2': eval2[category],
                'difference': {}
            }
            # Calculate difference for comparable metrics
            for metric in eval1[category]:
                if isinstance(eval1[category][metric], (int, float)):
                    comparison[category]['difference'][metric] = (
                        eval2[category][metric] - eval1[category][metric]
                    )
        
        return comparison