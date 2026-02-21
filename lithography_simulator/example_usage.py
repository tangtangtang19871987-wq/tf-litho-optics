"""Example usage of the DUV 193nm lithography simulator.

This example demonstrates the complete workflow of designing and optimizing
an immersion lithography process for the 7nm node, including:
- Setting up the lithography system
- Creating target patterns
- Applying OPC and ILT corrections
- Verifying the results
"""

import numpy as np
import tensorflow as tf
from lithography_simulator.core import LithographySystem
from lithography_simulator.mask import create_line_space_pattern, BinaryMask
from lithography_simulator.illumination import create_illumination
from lithography_simulator.mbopc import MBOPC
from lithography_simulator.ilt import ILT
from lithography_simulator.ml import MLSynthesizer, HybridOptimization
from lithography_simulator.verification import VerificationMetrics


def main():
    """Run the complete lithography simulation workflow."""
    print("Starting DUV 193nm Lithography Simulation Example")
    print("=" * 50)
    
    # 1. Set up the lithography system
    print("\n1. Setting up lithography system...")
    lithography_system = LithographySystem(
        wavelength=193e-9,  # 193 nm wavelength
        na=1.35,           # Numerical aperture
        n_immersion=1.44,  # Refractive index of immersion fluid
        image_scaling=4.0,  # 4x reduction
        shape=(128, 128),   # Simulation grid size
        resist_threshold=0.5  # Resist threshold
    )
    
    # 2. Create an illumination source
    print("\n2. Creating illumination source...")
    illumination = create_illumination(
        shape=(128, 128),
        source_type='partial_coherent',
        sigma=0.7,
        source_type='annular'
    )
    
    # 3. Create a target pattern (line-space pattern)
    print("\n3. Creating target pattern...")
    target_pattern = create_line_space_pattern(
        width=128, height=128, line_width=10, pitch=20
    )
    print(f"Target pattern shape: {target_pattern.shape}")
    
    # 4. Initialize a simple binary mask
    print("\n4. Initializing binary mask...")
    initial_mask = BinaryMask(shape=(128, 128))
    initial_mask.create_from_binary_pattern(target_pattern)
    
    # 5. Apply Model-Based OPC (MBOPC)
    print("\n5. Applying Model-Based OPC...")
    mbopc_optimizer = MBOPC(
        lithography_system=lithography_system,
        illumination=illumination,
        target_pattern=tf.constant(target_pattern, dtype=tf.complex64),
        learning_rate=0.01,
        max_iterations=50,
        regularization_weight=0.01
    )
    
    optimized_mask_mbopc, mbopc_history = mbopc_optimizer.run_optimization()
    
    # 6. Apply Inverse Lithography Technology (ILT)
    print("\n6. Applying Inverse Lithography Technology...")
    ilt_optimizer = ILT(
        lithography_system=lithography_system,
        illumination=illumination,
        target_pattern=tf.constant(target_pattern, dtype=tf.complex64),
        learning_rate=0.01,
        max_iterations=50,
        regularization_weights={
            'edge_smoothness': 0.01,
            'binary_enforcement': 0.1,
            'mask_area': 0.001
        }
    )
    
    optimized_mask_ilt, ilt_history = ilt_optimizer.run_optimization()
    
    # 7. Apply Machine Learning-based optimization
    print("\n7. Applying Machine Learning-based optimization...")
    # Create synthetic training data for demonstration
    training_targets = np.stack([target_pattern] * 10)
    training_masks = np.stack([target_pattern] * 10)  # In reality, this would be corrected masks
    
    ml_synthesizer = MLSynthesizer(
        model_type='unet',
        input_shape=(128, 128, 1),
        learning_rate=0.001
    )
    
    # Train the model (using synthetic data for this example)
    ml_synthesizer.train(
        training_targets=training_targets,
        training_masks=training_masks,
        epochs=5,
        batch_size=2
    )
    
    # Predict mask using ML
    predicted_mask_ml = ml_synthesizer.predict(target_pattern)
    
    # 8. Apply Hybrid Optimization (ML + Physics)
    print("\n8. Applying Hybrid Optimization (ML + Physics)...")
    hybrid_optimizer = HybridOptimization(
        ml_synthesizer=ml_synthesizer,
        lithography_system=lithography_system,
        illumination=illumination,
        learning_rate=0.001
    )
    
    hybrid_mask, hybrid_history = hybrid_optimizer.predict_and_refine(
        target_pattern, refinement_iterations=10
    )
    
    # 9. Verify the results using verification metrics
    print("\n9. Verifying results...")
    verifier = VerificationMetrics(pixel_size=1e-9)
    
    # Simulate the final aerial images for each approach
    mbopc_image = lithography_system.simulate_aerial_image(optimized_mask_mbopc.pattern)
    ilt_image = lithography_system.simulate_aerial_image(optimized_mask_ilt.pattern)
    ml_image = lithography_system.simulate_aerial_image(tf.constant(predicted_mask_ml, dtype=tf.complex64))
    hybrid_image = lithography_system.simulate_aerial_image(tf.constant(hybrid_mask, dtype=tf.complex64))
    
    # Evaluate each approach
    print("\nMBOPC Results:")
    mbopc_results = verifier.evaluate_mask_performance(
        lithography_system, optimized_mask_mbopc, target_pattern, mbopc_image.numpy()
    )
    print(f"  Process Window: {mbopc_results['process_window']['process_window_percent']:.2f}%")
    print(f"  Jaccard Index: {mbopc_results['pattern_fidelity']['jaccard_index']:.4f}")
    print(f"  CD Mean Error: {mbopc_results['critical_dimensions']['cd_mean']:.2e}m")
    
    print("\nILT Results:")
    ilt_results = verifier.evaluate_mask_performance(
        lithography_system, optimized_mask_ilt, target_pattern, ilt_image.numpy()
    )
    print(f"  Process Window: {ilt_results['process_window']['process_window_percent']:.2f}%")
    print(f"  Jaccard Index: {ilt_results['pattern_fidelity']['jaccard_index']:.4f}")
    print(f"  CD Mean Error: {ilt_results['critical_dimensions']['cd_mean']:.2e}m")
    
    print("\nML Results:")
    ml_results = verifier.evaluate_mask_performance(
        lithography_system, 
        BinaryMask((128, 128)).create_from_binary_pattern(predicted_mask_ml), 
        target_pattern, 
        ml_image.numpy()
    )
    print(f"  Process Window: {ml_results['process_window']['process_window_percent']:.2f}%")
    print(f"  Jaccard Index: {ml_results['pattern_fidelity']['jaccard_index']:.4f}")
    print(f"  CD Mean Error: {ml_results['critical_dimensions']['cd_mean']:.2e}m")
    
    print("\nHybrid Results:")
    hybrid_results = verifier.evaluate_mask_performance(
        lithography_system, 
        BinaryMask((128, 128)).create_from_binary_pattern(hybrid_mask), 
        target_pattern, 
        hybrid_image.numpy()
    )
    print(f"  Process Window: {hybrid_results['process_window']['process_window_percent']:.2f}%")
    print(f"  Jaccard Index: {hybrid_results['pattern_fidelity']['jaccard_index']:.4f}")
    print(f"  CD Mean Error: {hybrid_results['critical_dimensions']['cd_mean']:.2e}m")
    
    print("\n10. Complete lithography simulation workflow finished!")
    print("=" * 50)
    
    # Return results for further analysis if needed
    return {
        'mbopc': (optimized_mask_mbopc, mbopc_results),
        'ilt': (optimized_mask_ilt, ilt_results),
        'ml': (predicted_mask_ml, ml_results),
        'hybrid': (hybrid_mask, hybrid_results),
        'history': {
            'mbopc': mbopc_history,
            'ilt': ilt_history,
            'hybrid': hybrid_history
        }
    }


if __name__ == "__main__":
    results = main()