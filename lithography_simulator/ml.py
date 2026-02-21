"""Machine Learning module for DUV 193nm lithography simulator.

This module implements deep learning approaches for accelerating OPC and ILT computations,
including U-Net and Fourier Operator Network architectures for mask synthesis.
"""

import tensorflow as tf
import numpy as np
from typing import Tuple, Optional, Dict
from lithography_simulator.core import LithographySystem
from lithography_simulator.mask import Mask, BinaryMask
from lithography_simulator.illumination import IlluminationSource


class UNetMaskSynthesis(tf.keras.Model):
    """U-Net architecture for mask synthesis from target patterns."""
    
    def __init__(
        self,
        input_shape: Tuple[int, ...],
        filters: Tuple[int, ...] = (32, 64, 128, 256, 512),
        kernel_size: int = 3,
        dropout_rate: float = 0.2
    ):
        """
        Initialize the U-Net model for mask synthesis.
        
        Args:
            input_shape: Shape of the input (target pattern)
            filters: Number of filters for each layer
            kernel_size: Size of convolution kernels
            dropout_rate: Dropout rate for regularization
        """
        super(UNetMaskSynthesis, self).__init__()
        
        self.filters = filters
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        
        # Encoder (downsampling) layers
        self.encoder_layers = []
        for i, f in enumerate(filters):
            self.encoder_layers.append([
                tf.keras.layers.Conv2D(
                    f, kernel_size, padding='same', activation='relu',
                    name=f'encoder_conv_{i}_1'
                ),
                tf.keras.layers.Conv2D(
                    f, kernel_size, padding='same', activation='relu',
                    name=f'encoder_conv_{i}_2'
                ),
                tf.keras.layers.MaxPooling2D(2, name=f'encoder_pool_{i}'),
                tf.keras.layers.Dropout(dropout_rate, name=f'encoder_dropout_{i}')
            ])
        
        # Bottleneck layer
        self.bottleneck = [
            tf.keras.layers.Conv2D(
                filters[-1]*2, kernel_size, padding='same', activation='relu',
                name='bottleneck_conv_1'
            ),
            tf.keras.layers.Conv2D(
                filters[-1]*2, kernel_size, padding='same', activation='relu',
                name='bottleneck_conv_2'
            )
        ]
        
        # Decoder (upsampling) layers
        self.decoder_layers = []
        reversed_filters = list(reversed(filters))
        for i, f in enumerate(reversed_filters):
            self.decoder_layers.append([
                tf.keras.layers.Conv2D(
                    f, kernel_size, padding='same', activation='relu',
                    name=f'decoder_conv_{i}_1'
                ),
                tf.keras.layers.Conv2D(
                    f, kernel_size, padding='same', activation='relu',
                    name=f'decoder_conv_{i}_2'
                ),
                tf.keras.layers.UpSampling2D(2, name=f'decoder_upsample_{i}'),
                tf.keras.layers.Dropout(dropout_rate, name=f'decoder_dropout_{i}')
            ])
        
        # Final output layer
        self.output_layer = tf.keras.layers.Conv2D(
            1, 1, padding='same', activation='sigmoid', name='output_layer'
        )
        
        # Build the model by calling it with dummy input
        self._build_model(input_shape)
    
    def _build_model(self, input_shape):
        """Build the model by running a dummy forward pass."""
        dummy_input = tf.zeros((1,) + input_shape)
        _ = self(dummy_input)
    
    def call(self, inputs, training=None):
        """Forward pass of the U-Net model."""
        # Encoder path
        encoder_outputs = []
        x = inputs
        
        for layers in self.encoder_layers:
            for layer in layers[:-2]:  # All except pooling and dropout
                x = layer(x)
            encoder_outputs.append(x)  # Save for skip connections
            x = layers[-2](x)  # Pooling
            x = layers[-1](x)  # Dropout
        
        # Bottleneck
        for layer in self.bottleneck:
            x = layer(x)
        
        # Decoder path with skip connections
        for i, layers in enumerate(self.decoder_layers):
            x = layers[-2](x)  # Upsample
            x = tf.concat([x, encoder_outputs[-(i+1)]], axis=-1)  # Skip connection
            for layer in layers[:-2]:  # All except upsample and dropout
                x = layer(x)
            x = layers[-1](x)  # Dropout
        
        # Output layer
        output = self.output_layer(x)
        
        return output


class FourierOperatorNetwork(tf.keras.Model):
    """Fourier Operator Network for frequency-domain mask synthesis."""
    
    def __init__(
        self,
        input_shape: Tuple[int, ...],
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_fourier_layers: int = 2
    ):
        """
        Initialize the Fourier Operator Network.
        
        Args:
            input_shape: Shape of the input (target pattern)
            hidden_dim: Hidden dimension for spectral layers
            num_layers: Number of spectral processing layers
            num_fourier_layers: Number of Fourier operator layers
        """
        super(FourierOperatorNetwork, self).__init__()
        
        self.input_shape = input_shape
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_fourier_layers = num_fourier_layers
        
        # Input processing layer
        self.input_conv = tf.keras.layers.Conv2D(hidden_dim, 1, activation='relu')
        
        # Fourier operator layers
        self.fourier_layers = []
        for i in range(num_fourier_layers):
            self.fourier_layers.append(
                FourierOperatorLayer(hidden_dim)
            )
        
        # Spectral dense layers
        self.spectral_dense_layers = []
        for i in range(num_layers):
            self.spectral_dense_layers.append(
                SpectralDenseLayer(hidden_dim)
            )
        
        # Output processing layer
        self.output_conv = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')
    
    def call(self, inputs, training=None):
        """Forward pass of the Fourier Operator Network."""
        # Process input
        x = self.input_conv(inputs)
        
        # Apply Fourier operators
        for layer in self.fourier_layers:
            x = layer(x)
        
        # Apply spectral dense layers
        for layer in self.spectral_dense_layers:
            x = layer(x)
        
        # Generate output
        output = self.output_conv(x)
        
        return output


class FourierOperatorLayer(tf.keras.layers.Layer):
    """A layer that applies operations in the Fourier domain."""
    
    def __init__(self, hidden_dim: int, **kwargs):
        super(FourierOperatorLayer, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim
        
        # Learnable parameters for the Fourier domain transformation
        self.complex_weights_real = self.add_weight(
            shape=(hidden_dim,),
            initializer='random_normal',
            trainable=True,
            name='fourier_weights_real'
        )
        self.complex_weights_imag = self.add_weight(
            shape=(hidden_dim,),
            initializer='random_normal',
            trainable=True,
            name='fourier_weights_imag'
        )
        
        # Bias terms
        self.bias_real = self.add_weight(
            shape=(hidden_dim,),
            initializer='zeros',
            trainable=True,
            name='bias_real'
        )
        self.bias_imag = self.add_weight(
            shape=(hidden_dim,),
            initializer='zeros',
            trainable=True,
            name='bias_imag'
        )
    
    def call(self, inputs):
        """Apply the Fourier operator to the inputs."""
        # Reshape to separate channels for processing
        batch_size, height, width, channels = tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2], tf.shape(inputs)[3]
        
        # Pad if necessary to power of 2 for efficient FFT
        h_pad = tf.math.pow(2, tf.math.ceil(tf.math.log(tf.cast(height, tf.float32)) / tf.math.log(2.0)))
        w_pad = tf.math.pow(2, tf.math.ceil(tf.math.log(tf.cast(width, tf.float32)) / tf.math.log(2.0)))
        h_pad = tf.cast(h_pad, tf.int32)
        w_pad = tf.cast(w_pad, tf.int32)
        
        inputs_padded = tf.image.resize_with_pad(inputs, h_pad, w_pad)
        
        # Transform to Fourier domain channel by channel
        fft_inputs = tf.signal.fft2d(tf.cast(inputs_padded, tf.complex64))
        
        # Apply learnable transformation in Fourier domain
        # Create complex weights
        complex_weights = tf.complex(self.complex_weights_real, self.complex_weights_imag)
        bias = tf.complex(self.bias_real, self.bias_imag)
        
        # Apply transformation
        transformed_fft = fft_inputs * complex_weights + bias
        
        # Transform back to spatial domain
        outputs_padded = tf.cast(tf.signal.ifft2d(transformed_fft), tf.float32)
        
        # Crop back to original size
        outputs = outputs_padded[:, :height, :width, :]
        
        return outputs


class SpectralDenseLayer(tf.keras.layers.Layer):
    """A dense layer applied in the spectral domain."""
    
    def __init__(self, hidden_dim: int, **kwargs):
        super(SpectralDenseLayer, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim
        
        # Dense layer for spectral processing
        self.dense = tf.keras.layers.Dense(hidden_dim, activation='relu')
    
    def call(self, inputs):
        """Apply dense transformation in spectral domain."""
        # Flatten spatial dimensions
        batch_size = tf.shape(inputs)[0]
        height, width, channels = inputs.shape[1], inputs.shape[2], inputs.shape[3]
        
        # Reshape to (batch, height*width, channels) for dense processing
        reshaped = tf.reshape(inputs, [batch_size, height * width, channels])
        
        # Apply dense layer
        processed = self.dense(reshaped)
        
        # Reshape back to (batch, height, width, channels)
        outputs = tf.reshape(processed, [batch_size, height, width, self.hidden_dim])
        
        return outputs


class MLSynthesizer:
    """ML-based mask synthesizer using U-Net or Fourier Operator Network."""
    
    def __init__(
        self,
        model_type: str = 'unet',  # 'unet' or 'fourier'
        input_shape: Tuple[int, int, int] = (128, 128, 1),
        learning_rate: float = 0.001
    ):
        """
        Initialize the ML synthesizer.
        
        Args:
            model_type: Type of model ('unet' or 'fourier')
            input_shape: Shape of the input (height, width, channels)
            learning_rate: Learning rate for training
        """
        self.model_type = model_type
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        
        # Create the appropriate model
        if model_type == 'unet':
            self.model = UNetMaskSynthesis(input_shape)
        elif model_type == 'fourier':
            self.model = FourierOperatorNetwork(input_shape)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Compile the model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        self.training_history = []
    
    def preprocess_target(self, target_pattern: np.ndarray) -> np.ndarray:
        """
        Preprocess the target pattern for input to the network.
        
        Args:
            target_pattern: Target pattern as numpy array
            
        Returns:
            Preprocessed target pattern
        """
        # Ensure the target is the right shape
        if target_pattern.shape != self.input_shape:
            # Resize if necessary
            target_pattern = tf.image.resize(
                target_pattern[..., np.newaxis] if len(target_pattern.shape) == 2 else target_pattern,
                self.input_shape[:2]
            ).numpy()
        
        # Normalize to [0, 1]
        target_pattern = (target_pattern - np.min(target_pattern)) / (np.max(target_pattern) - np.min(target_pattern))
        
        # Add batch dimension if not present
        if len(target_pattern.shape) == len(self.input_shape):
            return target_pattern[np.newaxis, ...]
        else:
            return target_pattern
    
    def postprocess_output(self, output_mask: np.ndarray) -> np.ndarray:
        """
        Postprocess the network output to create a valid mask.
        
        Args:
            output_mask: Output from the network
            
        Returns:
            Postprocessed mask
        """
        # Remove batch dimension if present
        if len(output_mask.shape) == len(self.input_shape) + 1:
            output_mask = output_mask[0]
        
        # Threshold to create binary mask
        binary_mask = (output_mask > 0.5).astype(np.float32)
        
        return binary_mask
    
    def train(
        self,
        training_targets: np.ndarray,
        training_masks: np.ndarray,
        validation_targets: Optional[np.ndarray] = None,
        validation_masks: Optional[np.ndarray] = None,
        epochs: int = 50,
        batch_size: int = 8
    ) -> Dict[str, list]:
        """
        Train the ML model.
        
        Args:
            training_targets: Training target patterns
            training_masks: Training mask patterns (ground truth)
            validation_targets: Validation target patterns
            validation_masks: Validation mask patterns
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Training history
        """
        print(f"Training {self.model_type} model...")
        
        # Preprocess inputs
        x_train = self.preprocess_target(training_targets)
        y_train = self.preprocess_target(training_masks)
        
        # Prepare validation data if provided
        validation_data = None
        if validation_targets is not None and validation_masks is not None:
            x_val = self.preprocess_target(validation_targets)
            y_val = self.preprocess_target(validation_masks)
            validation_data = (x_val, y_val)
        
        # Train the model
        history = self.model.fit(
            x_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        # Store training history
        self.training_history = history.history
        
        print(f"{self.model_type} model training completed.")
        return self.training_history
    
    def predict(self, target_pattern: np.ndarray) -> np.ndarray:
        """
        Predict a mask for a given target pattern.
        
        Args:
            target_pattern: Target pattern to synthesize a mask for
            
        Returns:
            Synthesized mask
        """
        # Preprocess the input
        input_pattern = self.preprocess_target(target_pattern)
        
        # Run inference
        output = self.model(input_pattern, training=False)
        
        # Postprocess the output
        mask = self.postprocess_output(output.numpy())
        
        return mask


class HybridOptimization:
    """Hybrid approach combining ML prediction with physics-based refinement."""
    
    def __init__(
        self,
        ml_synthesizer: MLSynthesizer,
        lithography_system: LithographySystem,
        illumination: IlluminationSource,
        learning_rate: float = 0.001
    ):
        """
        Initialize the hybrid optimization approach.
        
        Args:
            ml_synthesizer: Trained ML model for initial mask prediction
            lithography_system: Lithography system for physics simulation
            illumination: Illumination source
            learning_rate: Learning rate for physics-based refinement
        """
        self.ml_synthesizer = ml_synthesizer
        self.lithography_system = lithography_system
        self.illumination = illumination
        self.learning_rate = learning_rate
    
    def predict_and_refine(
        self,
        target_pattern: np.ndarray,
        refinement_iterations: int = 10
    ) -> Tuple[np.ndarray, Dict[str, list]]:
        """
        Predict an initial mask using ML and refine using physics-based optimization.
        
        Args:
            target_pattern: Target pattern to synthesize a mask for
            refinement_iterations: Number of refinement iterations
            
        Returns:
            Tuple of (refined mask, refinement history)
        """
        print("Starting hybrid ML-physics optimization...")
        
        # Step 1: Get initial prediction from ML model
        initial_mask = self.ml_synthesizer.predict(target_pattern)
        
        # Convert to complex tensor for physics simulation
        initial_mask_tensor = tf.cast(initial_mask, tf.complex64)
        
        # Step 2: Physics-based refinement using gradient descent
        refined_mask = tf.Variable(initial_mask_tensor, dtype=tf.complex64)
        
        refinement_history = []
        
        for iteration in range(refinement_iterations):
            with tf.GradientTape() as tape:
                tape.watch(refined_mask)
                
                # Simulate the current mask
                aerial_image = self.lithography_system.simulate_aerial_image(refined_mask)
                
                # Calculate loss (difference from target)
                loss = tf.reduce_mean(tf.square(aerial_image - target_pattern))
            
            # Calculate gradients
            gradients = tape.gradient(loss, refined_mask)
            
            # Update mask using gradient descent
            refined_mask.assign_sub(self.learning_rate * gradients)
            
            # Enforce constraints (binary mask)
            magnitude = tf.abs(refined_mask)
            phase = tf.math.angle(refined_mask)
            
            # Constrain magnitude to be binary-like
            magnitude = tf.nn.sigmoid(10.0 * (magnitude - 0.5))
            
            # Reconstruct complex mask
            refined_mask.assign(magnitude * tf.exp(tf.complex(tf.zeros_like(phase), phase)))
            
            # Record loss
            refinement_history.append({
                'iteration': iteration,
                'loss': loss.numpy()
            })
            
            if iteration % 5 == 0:
                print(f"Refinement iteration {iteration}, Loss: {loss.numpy():.6f}")
        
        print("Hybrid optimization completed.")
        
        return refined_mask.numpy(), refinement_history