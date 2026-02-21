"""Lithography Simulator package for DUV 193nm immersion lithography.

This package provides comprehensive tools for simulating and optimizing
immersion lithography processes targeting the 7nm node, including
core simulation, mask modeling, illumination, optics, OPC, ILT, ML acceleration,
and verification modules.
"""

# Core modules
from .core import LithographySystem
from .mask import Mask, BinaryMask, PhaseShiftMask
from .illumination import (
    IlluminationSource,
    CoherentSource,
    PartialCoherentSource,
    VectorialPolarizationSource,
    create_illumination
)
from .optics import (
    ProjectionLens,
    VectorialImagingModel,
    HopkinsImagingModel
)
from .mbopc import MBOPC, EdgePlacementErrorCalculator, ProcessWindowOptimizer
from .ilt import ILT, PixelBasedILT, SourceMaskOptimization
from .ml import (
    UNetMaskSynthesis,
    FourierOperatorNetwork,
    MLSynthesizer,
    HybridOptimization
)
from .verification import (
    calculate_critical_dimensions,
    calculate_line_edge_roughness,
    calculate_overlap_area,
    calculate_process_window,
    calculate_mask_error_factor,
    calculate_resolution_limits,
    comprehensive_verification_report,
    VerificationMetrics
)

__version__ = "1.0.0"
__author__ = "Lithography Simulator Team"
__all__ = [
    # Core
    'LithographySystem',
    
    # Mask
    'Mask', 'BinaryMask', 'PhaseShiftMask',
    
    # Illumination
    'IlluminationSource', 'CoherentSource', 'PartialCoherentSource',
    'VectorialPolarizationSource', 'create_illumination',
    
    # Optics
    'ProjectionLens', 'VectorialImagingModel', 'HopkinsImagingModel',
    
    # MBOPC
    'MBOPC', 'EdgePlacementErrorCalculator', 'ProcessWindowOptimizer',
    
    # ILT
    'ILT', 'PixelBasedILT', 'SourceMaskOptimization',
    
    # ML
    'UNetMaskSynthesis', 'FourierOperatorNetwork', 'MLSynthesizer', 'HybridOptimization',
    
    # Verification
    'calculate_critical_dimensions', 'calculate_line_edge_roughness', 
    'calculate_overlap_area', 'calculate_process_window', 'calculate_mask_error_factor',
    'calculate_resolution_limits', 'comprehensive_verification_report', 'VerificationMetrics'
]