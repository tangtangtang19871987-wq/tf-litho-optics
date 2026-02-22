# DUV 193nm Lithography Simulator for 7nm Node

This repository contains a comprehensive Deep Ultraviolet (DUV) lithography simulation system designed specifically for 7nm technology node processes. The system includes multiple optimization algorithms and verification tools to simulate and optimize photomask patterns for advanced semiconductor manufacturing.

## Features

- **Physics-based Simulation**: Full electromagnetic simulation of DUV lithography process at 193nm wavelength
- **Multiple OPC Algorithms**: 
  - Model-Based OPC (MBOPC)
  - Inverse Lithography Technology (ILT)
  - Machine Learning-based optimization
  - Hybrid ML-physics optimization
- **Process Window Analysis**: Comprehensive verification metrics
- **Advanced Verification**: Including CD uniformity, pattern fidelity, and process robustness metrics

## System Specifications

- **Wavelength**: 193 nm (ArF Excimer Laser)
- **Numerical Aperture (NA)**: 1.35
- **Coherent Factor (σ)**: 0.7
- **Target Node**: 7nm technology
- **Mask Type**: Advanced Binary Mask with EUV-like absorber properties
- **Focus Range**: ±50nm around best focus
- **Exposure Dose Range**: 15-25 mJ/cm²

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the complete example:

```python
python lithography_simulator/example_usage.py
```

This will:
1. Initialize the lithography system
2. Generate target patterns
3. Apply multiple optimization techniques (MBOPC, ILT, ML, Hybrid)
4. Perform comprehensive verification
5. Output performance metrics

## Project Structure

- `core.py`: Main lithography system implementation
- `mask.py`: Photomask representation and manipulation
- `illumination.py`: Illumination source modeling
- `optics.py`: Optical system modeling
- `mbopc.py`: Model-Based OPC optimizer
- `ilt.py`: Inverse Lithography Technology optimizer
- `ml.py`: Machine learning-based optimization
- `verification.py`: Verification and validation tools
- `data_generation.py`: Training data generation for ML models
- `training_scripts.py`: Scripts for training ML models

## Key Components

### Lithography System
- Coherent and partially coherent illumination
- Vectorial optical modeling
- Resist process modeling
- Noise modeling for realistic simulations

### Optimization Algorithms
1. **MBOPC**: Gradient-based optimization for mask correction
2. **ILT**: Pixel-based inverse optimization
3. **ML**: U-Net based mask synthesis
4. **Hybrid**: ML-physics combined optimization

### Verification Framework
- Process window analysis (focus/exposure latitude)
- Pattern fidelity metrics (Jaccard index, CD errors)
- Statistical verification across multiple patterns

## Performance Metrics

The system evaluates performance using:
- **Jaccard Index**: Pattern similarity metric
- **CD Mean Error**: Critical dimension accuracy
- **Process Window**: Robustness to focus/exposure variations
- **Pattern Fidelity**: Edge placement error

## Applications

This simulator is suitable for:
- Photomask optimization research
- OPC algorithm development
- Lithography process development
- Semiconductor manufacturing process optimization
- Academic research in computational lithography

## Note

This is a research-grade simulator designed for advanced lithography process development. The model parameters are calibrated for 7nm node processes but can be adapted for other technology nodes.

## License

MIT License - see LICENSE file for details.