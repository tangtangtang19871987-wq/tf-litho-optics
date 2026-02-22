"""
Training Scripts for DUV Lithography Simulation Models

This module contains training scripts for U-Net models
to predict aerial images from mask layouts in DUV lithography.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import json


def load_training_data(filepath='duv_litho_training_data.npz'):
    """
    Load the generated training data.
    
    Args:
        filepath: Path to the training data file
        
    Returns:
        tuple: (masks, aerial_images) as numpy arrays
    """
    data = np.load(filepath)
    X = data['X']  # Masks
    Y = data['Y']  # Aerial images
    
    return X, Y


def normalize_data(X, Y):
    """
    Normalize the input data.
    
    Args:
        X: Input masks
        Y: Target aerial images
        
    Returns:
        tuple: Normalized (X, Y)
    """
    X_norm = (X - X.min()) / (X.max() - X.min())
    Y_norm = (Y - Y.min()) / (Y.max() - Y.min())
    
    return X_norm, Y_norm


class UNetModel(keras.Model):
    """
    U-Net model for predicting aerial images from mask layouts.
    """
    
    def __init__(self, input_shape=(256, 256, 1)):
        super(UNetModel, self).__init__()
        self.input_shape_param = input_shape
        
        # Encoder
        self.conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')
        self.conv2 = layers.Conv2D(64, 3, activation='relu', padding='same')
        self.pool1 = layers.MaxPooling2D(2)
        
        self.conv3 = layers.Conv2D(128, 3, activation='relu', padding='same')
        self.conv4 = layers.Conv2D(128, 3, activation='relu', padding='same')
        self.pool2 = layers.MaxPooling2D(2)
        
        self.conv5 = layers.Conv2D(256, 3, activation='relu', padding='same')
        self.conv6 = layers.Conv2D(256, 3, activation='relu', padding='same')
        self.pool3 = layers.MaxPooling2D(2)
        
        self.conv7 = layers.Conv2D(512, 3, activation='relu', padding='same')
        self.conv8 = layers.Conv2D(512, 3, activation='relu', padding='same')
        self.drop1 = layers.Dropout(0.5)
        self.pool4 = layers.MaxPooling2D(2)
        
        # Bottleneck
        self.conv9 = layers.Conv2D(1024, 3, activation='relu', padding='same')
        self.conv10 = layers.Conv2D(1024, 3, activation='relu', padding='same')
        self.drop2 = layers.Dropout(0.5)
        
        # Decoder
        self.upconv1 = layers.Conv2DTranspose(512, 2, strides=2, activation='relu', padding='same')
        self.dconv1 = layers.Conv2D(512, 3, activation='relu', padding='same')
        self.dconv2 = layers.Conv2D(512, 3, activation='relu', padding='same')
        
        self.upconv2 = layers.Conv2DTranspose(256, 2, strides=2, activation='relu', padding='same')
        self.dconv3 = layers.Conv2D(256, 3, activation='relu', padding='same')
        self.dconv4 = layers.Conv2D(256, 3, activation='relu', padding='same')
        
        self.upconv3 = layers.Conv2DTranspose(128, 2, strides=2, activation='relu', padding='same')
        self.dconv5 = layers.Conv2D(128, 3, activation='relu', padding='same')
        self.dconv6 = layers.Conv2D(128, 3, activation='relu', padding='same')
        
        self.upconv4 = layers.Conv2DTranspose(64, 2, strides=2, activation='relu', padding='same')
        self.dconv7 = layers.Conv2D(64, 3, activation='relu', padding='same')
        self.dconv8 = layers.Conv2D(64, 3, activation='relu', padding='same')
        
        self.output_layer = layers.Conv2D(1, 1, activation='sigmoid', padding='same')
    
    def call(self, inputs):
        # Encoder
        c1 = self.conv1(inputs)
        c1 = self.conv2(c1)
        p1 = self.pool1(c1)
        
        c2 = self.conv3(p1)
        c2 = self.conv4(c2)
        p2 = self.pool2(c2)
        
        c3 = self.conv5(p2)
        c3 = self.conv6(c3)
        p3 = self.pool3(c3)
        
        c4 = self.conv7(p3)
        c4 = self.conv8(c4)
        p4 = self.pool4(c4)
        
        # Bottleneck
        c5 = self.conv9(p4)
        c5 = self.conv10(c5)
        d5 = self.drop1(c5)
        
        # Decoder
        u6 = self.upconv1(d5)
        u6 = tf.concat([u6, c4], axis=-1)
        c6 = self.dconv1(u6)
        c6 = self.dconv2(c6)
        
        u7 = self.upconv2(c6)
        u7 = tf.concat([u7, c3], axis=-1)
        c7 = self.dconv3(u7)
        c7 = self.dconv4(c7)
        
        u8 = self.upconv3(c7)
        u8 = tf.concat([u8, c2], axis=-1)
        c8 = self.dconv5(u8)
        c8 = self.dconv6(c8)
        
        u9 = self.upconv4(c8)
        u9 = tf.concat([u9, c1], axis=-1)
        c9 = self.dconv7(u9)
        c9 = self.dconv8(c9)
        
        outputs = self.output_layer(c9)
        
        return outputs


def train_unet(X_train, Y_train, X_val, Y_val, epochs=50, batch_size=4):
    """
    Train the U-Net model.
    
    Args:
        X_train: Training masks
        Y_train: Training aerial images
        X_val: Validation masks
        Y_val: Validation aerial images
        epochs: Number of training epochs
        batch_size: Batch size for training
        
    Returns:
        Trained U-Net model
    """
    # Reshape data for TensorFlow
    X_train = X_train.reshape(-1, 256, 256, 1)
    Y_train = Y_train.reshape(-1, 256, 256, 1)
    X_val = X_val.reshape(-1, 256, 256, 1)
    Y_val = Y_val.reshape(-1, 256, 256, 1)
    
    # Create and compile model
    model = UNetModel()
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    # Callbacks
    early_stopping = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=5, min_lr=1e-7)
    
    # Train the model
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    return model, history


def evaluate_model(model, X_test, Y_test, model_name="Model"):
    """
    Evaluate the trained model.
    
    Args:
        model: Trained model
        X_test: Test masks
        Y_test: Test aerial images
        model_name: Name of the model for reporting
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Reshape for TensorFlow
    X_test_tf = X_test.reshape(-1, 256, 256, 1)
    Y_pred = model.predict(X_test_tf)
    Y_pred = Y_pred.reshape(Y_test.shape)
    
    # Calculate metrics
    mse = mean_squared_error(Y_test.flatten(), Y_pred.flatten())
    mae = mean_absolute_error(Y_test.flatten(), Y_pred.flatten())
    rmse = np.sqrt(mse)
    
    # Calculate R² score
    ss_res = np.sum((Y_test - Y_pred) ** 2)
    ss_tot = np.sum((Y_test - np.mean(Y_test)) ** 2)
    r2_score = 1 - (ss_res / ss_tot)
    
    metrics = {
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'R² Score': r2_score
    }
    
    print(f"{model_name} Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.6f}")
    
    return metrics, Y_pred


def save_results_csv(metrics, filename='model_comparison.csv'):
    """
    Save model results to CSV.
    
    Args:
        metrics: Metrics for model
        filename: Output CSV filename
    """
    # Prepare data for DataFrame
    data = {
        'Metric': ['MSE', 'MAE', 'RMSE', 'R² Score'],
        'U-Net': [
            metrics['MSE'],
            metrics['MAE'],
            metrics['RMSE'],
            metrics['R² Score']
        ]
    }
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")
    
    return df


def main():
    """
    Main function to run the training and evaluation pipeline.
    """
    print("Loading training data...")
    X, Y = load_training_data()
    
    print(f"Data shapes - Masks: {X.shape}, Aerial Images: {Y.shape}")
    
    # Normalize data
    X, Y = normalize_data(X, Y)
    
    # Split data into train/validation/test sets
    n_samples = X.shape[0]
    n_train = int(0.7 * n_samples)
    n_val = int(0.15 * n_samples)
    
    X_train, Y_train = X[:n_train], Y[:n_train]
    X_val, Y_val = X[n_train:n_train+n_val], Y[n_train:n_train+n_val]
    X_test, Y_test = X[n_train+n_val:], Y[n_train+n_val:]
    
    print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    
    # Train U-Net model
    print("\nTraining U-Net model...")
    unet_model, unet_history = train_unet(X_train, Y_train, X_val, Y_val)
    
    # Evaluate model
    print("\nEvaluating U-Net model...")
    unet_metrics, unet_predictions = evaluate_model(unet_model, X_test, Y_test, "U-Net")
    
    # Save comparison results
    df_results = save_results_csv(unet_metrics)
    print("\nModel Results:")
    print(df_results)
    
    # Save model
    unet_model.save('unet_model.h5')
    
    print("\nTraining and evaluation completed!")
    print("Model saved as 'unet_model.h5'")
    print("Results saved as 'model_comparison.csv'")


if __name__ == "__main__":
    main()
