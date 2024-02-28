# Import necessary libraries for data handling, machine learning, and plotting.
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.initializers import RandomNormal

# Block 1
# Define a subclass of tensorflow.keras.models.Model to implement a custom model with Newton Optimization.
class NewtonOptimizedModel(Model):
    def __init__(self):
        # Initialize the parent class constructor to properly setup the model.
        super(NewtonOptimizedModel, self).__init__()
        # Define the model layers with specific configurations.
        # First dense layer with 15 neurons and tanh activation function.
        self.dense = Dense(15, activation='tanh', input_shape=(4,), kernel_initializer=RandomNormal())
        # Second dense layer with 10 neurons and tanh activation function.
        self.dense1 = Dense(10, activation='tanh', kernel_initializer=RandomNormal())
        # Output layer with 3 neurons (for classification into 3 classes) with softmax activation function.
        self.output_layer = Dense(3, activation='softmax', kernel_initializer=RandomNormal())

    # Define the forward pass through the network.
    def call(self, inputs):
        x = self.dense(inputs)  # Pass inputs through the first layer.
        x = self.dense1(x)  # Pass the result through the second layer.
        return self.output_layer(x)  # Return the output of the final layer.

    # Regularize the Hessian matrix to avoid inversion problems by adding a small value to its diagonal.
    def regularize_hessian(self, hessian, regularization_strength=1e-9):
        regularized_hessian = hessian + tf.eye(tf.shape(hessian)[0]) * regularization_strength
        return regularized_hessian

    # Override the train_step method to implement custom training logic using Newton's method.
    def train_step(self, data, subsampling_parameter=0.5):
        x, y = data  # Unpack the data.

        # Use GradientTape to record operations for automatic differentiation.
        with tf.GradientTape(persistent=True) as tape:
            y_pred = self(x, training=True)  # Make predictions.
            loss = self.compiled_loss(y, y_pred)  # Compute loss.

        layer_gradients = {}  # Dictionary to store gradients for each layer.
        layer_hessians = {}  # Dictionary to store Hessians for each layer.

        for layer in self.layers:
            with tf.GradientTape(persistent=True) as hessian_tape:
                # Recompute the loss for Hessian computation.
                layer_loss = self.compiled_loss(y, self(x, training=True))
                grads = hessian_tape.gradient(layer_loss, layer.trainable_variables)

                # Decide whether to subsample the layer based on a random threshold.
                if np.random.rand() < subsampling_parameter:
                    # Compute and store Hessians for subsampled layers.
                    hessians = [hessian_tape.jacobian(grad, var) for grad, var in zip(grads, layer.trainable_variables)]
                    layer_hessians[layer.name] = hessians
                else:
                    # For non-sampled layers, store the gradients.
                    layer_gradients[layer.name] = grads

        # Update the model parameters using either gradients or Hessians.
        for layer in self.layers:
            if layer.name in layer_hessians:
                for hessian, grad, var in zip(layer_hessians[layer.name], layer_gradients.get(layer.name, []), layer.trainable_variables):
                    # Flatten Hessian, regularize, compute its inverse, and apply the update.
                    hessian_flat = tf.reshape(hessian, [tf.size(var), -1])
                    reg_hessian = self.regularize_hessian(hessian_flat)
                    inv_hessian = tf.linalg.inv(reg_hessian)
                    grad_flat = tf.reshape(grad, [-1, 1])
                    var_update = tf.linalg.matmul(inv_hessian, grad_flat)
                    var.assign_sub(tf.reshape(var_update, var.shape))
            elif layer.name in layer_gradients:
                # For non-sampled layers, update the parameters using gradients directly.
                for grad, var in zip(layer_gradients[layer.name], layer.trainable_variables):
                    var.assign_sub(grad)

        # Clean up the GradientTape resources.
        del tape
        # Update the state of metrics.
        self.compiled_metrics.update_state(y, y_pred)
        # Return the results of metric computation.
        return {m.name: m.result() for m in self.metrics}
