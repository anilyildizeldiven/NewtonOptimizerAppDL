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

# Define a custom model class 
class NewtonOptimizedModel(Model):
    # Initialize the Model & define the layers
    # Change layers, num_classes and the subsampling_rate according to your needs
    def __init__(self, input_shape=(4,), subsampling_rate=0.9, num_classes=3):
        super(NewtonOptimizedModel, self).__init__()
        self.dense = Dense(15, activation='tanh', input_shape=input_shape, kernel_initializer=RandomNormal())
        self.dense1 = Dense(10, activation='tanh', kernel_initializer=RandomNormal())
        self.output_layer = Dense(num_classes, activation='softmax', kernel_initializer=RandomNormal())
        self.subsampling_rate = subsampling_rate
        self.last_subsampled_indices = [] 

    def call(self, inputs):
        x = self.dense(inputs)
        x = self.dense1(x)
        return self.output_layer(x)
        
    # Helper method to regularize the Hessian matrix by adding a small value to its diagonal
    # Change regularization_strength if necessary
    def regularize_hessian(self, hessian, regularization_strength=1e-2):
        n = tf.shape(hessian)[-1]
        return hessian + tf.eye(n) * regularization_strength
        
    # Custom training step that incorporates Newton's method optimization aspects
    def train_step(self, data):
        x, y = data
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.trainable_variables)
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred)
            
        # Select a subset of variables to compute the Hessian for, based on the subsampling rate
        subsampled_indices = np.random.choice(range(len(self.trainable_variables)), 
                                              size=int(np.floor(len(self.trainable_variables) * self.subsampling_rate)), 
                                              replace=False)
        self.last_subsampled_indices = np.random.choice(range(len(self.trainable_variables)), 
                                                        size=int(np.floor(len(self.trainable_variables) * self.subsampling_rate)), 
                                                        replace=False) 
        update_norms = []
        
         # Iterate over each trainable variable & compute gradients 
        for i, var in enumerate(self.trainable_variables):
            with tf.GradientTape() as hessian_tape:
                hessian_tape.watch(var)
                with tf.GradientTape() as grad_tape:
                    grad_tape.watch(var)
                    y_pred_inner = self(x, training=True)
                    loss_inner = self.compiled_loss(y, y_pred_inner)
                grad = grad_tape.gradient(loss_inner, var)
                
            # Compute the Hessian, the respective inversion and the newton method update step, as well as the update norm for subsampled variables
            if i in subsampled_indices:
                hessian = hessian_tape.jacobian(grad, var)
                var_size = tf.reduce_prod(var.shape)
                hessian_flat = tf.reshape(hessian, [var_size, var_size])
                hessian_reg = self.regularize_hessian(hessian_flat)
                inv_hessian = tf.linalg.inv(hessian_reg + tf.eye(var_size))
                update = tf.linalg.matmul(inv_hessian, tf.reshape(grad, [-1, 1]))
                update_reshaped = tf.reshape(update, var.shape)
                var.assign_sub(update_reshaped)
                update_norm = tf.norm(inv_hessian)
                update_norms.append(update_norm)
                
        # Compute average update norm for subsampled variables
        avg_update_norm = tf.reduce_mean(update_norms) if update_norms else 0

        # Apply scaled gradient update to non-sampled variables using the average update norm
        for i, var in enumerate(self.trainable_variables):
            if i not in subsampled_indices:
                with tf.GradientTape() as grad_tape:
                    grad_tape.watch(var)
                    y_pred_inner = self(x, training=True)
                    loss_inner = self.compiled_loss(y, y_pred_inner)
                grad = grad_tape.gradient(loss_inner, var)
                scaled_update = grad * avg_update_norm
                var.assign_sub(scaled_update)

        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}
