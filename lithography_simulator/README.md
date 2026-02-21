# DUV 193nm Lithography Simulator for 7nm Node

## Overview
This lithography simulator is designed for Deep Ultraviolet (DUV) 193nm immersion lithography targeting the 7nm technology node. The simulator focuses on 2D aerial image simulation with advanced Model-Based OPC (MBOPC) and Inverse Lithography Technology (ILT) capabilities.

## Key Features
- 2D aerial image simulation using vectorial optical models
- Support for immersion lithography (n=1.44, effective λ≈134nm)
- Model-based OPC with gradient optimization
- Inverse lithography technology (ILT) for mask synthesis
- U-Net and Fourier Operator Network for ML-based optimization
- Comprehensive verification metrics for 7nm node patterns

## Technical Specifications
- **Wavelength**: 193nm (DUV) with immersion (effective λ ≈ 134nm)
- **Numerical Aperture**: 1.35 (immersion)
- **Simulation Focus**: 2D aerial image only
- **OPC Approach**: Model-based and inverse lithography
- **ML Models**: U-Net and Fourier Operator Network

## Architecture
The simulator consists of the following modules:
1. Core simulation engine using TensorFlow Torchoptics
2. 2D mask representation and manipulation
3. DUV-specific illumination models
4. Vectorial optical imaging with partial coherence
5. Model-based OPC and ILT optimization
6. Machine learning acceleration
7. Verification and metrics

## Requirements
- Python 3.8+
- TensorFlow 2.x
- NumPy, SciPy, scikit-image
- TensorFlow Torchoptics (already implemented)

## Installation
```bash
pip install -r requirements.txt
```

## Usage
Refer to the example notebooks for usage instructions.