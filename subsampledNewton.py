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

# Define a subclass of tensorflow.keras.models.Model to implement a custom model with Newton Optimization.
class NewtonOptimizedModel(Model):
    def __init__(self): #Define Layers; Here: Examplifies two dense Layers and one output Layer
        super(NewtonOptimizedModel, self).__init__()
        self.dense = Dense(15, activation='tanh', input_shape=(4,), kernel_initializer=RandomNormal())
        self.dense1 = Dense(10, activation='tanh', kernel_initializer=RandomNormal())
        self.output_layer = Dense(3, activation='softmax', kernel_initializer=RandomNormal())

    # Forward pass: Sequentially pass inputs through dense1 -> dense2 -> output_layer.
    def call(self, inputs):
        x = self.dense(inputs)  
        x = self.dense1(x)  
        return self.output_layer(x)  

    # Add a small constant to the diagonal of the Hessian to ensure it's invertible.
    # Change regularization_strength if needed / depending on the problem
    def regularize_hessian(self, hessian, regularization_strength=1e-4):
        regularized_hessian = hessian + tf.eye(tf.shape(hessian)[0]) * regularization_strength
        return regularized_hessian

    # Override the train_step method to implement custom training logic using Newton's method.
    def train_step(self, data, subsampling_parameter=0.5): # subsampling_parameter defined between 0 and 1
        # Change subsampling_parameter to 1 for full newton method updates or to 0 for full gradient descent updates
        # Change subsampling_parameter if needed / depending on the problem
        x, y = data  # Unpack the data.

        with tf.GradientTape(persistent=True) as tape:
            y_pred = self(x, training=True)  # Make predictions.
            loss = self.compiled_loss(y, y_pred)  # Compute loss.
        
        # Initialize storage for gradients and Hessians.
        layer_gradients = {}
        layer_hessians = {} 

        for layer in self.layers:
            # Use a nested GradientTape for Hessian computation.

            with tf.GradientTape(persistent=True) as hessian_tape:
                with tf.GradientTape() as t1:
                    y_pred = self(x, training=True)
                    loss = self.compiled_loss(y, y_pred)
                    # Compute gradients of loss w.r.t. layer variables.

                grads = t1.gradient(loss, layer.trainable_variables)
                

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
                # For non-sampled layers use gradients to calculate the update step
                for grad, var in zip(layer_gradients[layer.name], layer.trainable_variables):
                    var.assign_sub(grad)  

        del tape
        # Update the state of metrics & return the results of metric computation.
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}
