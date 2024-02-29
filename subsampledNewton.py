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

class NewtonOptimizedModel(Model):
    def __init__(self, input_shape=(4,), subsampling_rate=0.9, num_classes=3):
        super(NewtonOptimizedModel, self).__init__()
        self.dense = Dense(15, activation='tanh', input_shape=input_shape, kernel_initializer=RandomNormal())
        self.dense1 = Dense(10, activation='tanh', kernel_initializer=RandomNormal())
        self.output_layer = Dense(num_classes, activation='softmax', kernel_initializer=RandomNormal())
        self.subsampling_rate = subsampling_rate

    def call(self, inputs):
        x = self.dense(inputs)
        x = self.dense1(x)
        return self.output_layer(x)

    def regularize_hessian(self, hessian, regularization_strength=1e-2):
        n = tf.shape(hessian)[-1]
        return hessian + tf.eye(n) * regularization_strength

    def train_step(self, data):
        x, y = data
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.trainable_variables)
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred)

        subsampled_indices = np.random.choice(range(len(self.trainable_variables)), 
                                              size=int(np.floor(len(self.trainable_variables) * self.subsampling_rate)), 
                                              replace=False)
        update_norms = []

        for i, var in enumerate(self.trainable_variables):
            with tf.GradientTape() as hessian_tape:
                hessian_tape.watch(var)
                with tf.GradientTape() as grad_tape:
                    grad_tape.watch(var)
                    y_pred_inner = self(x, training=True)
                    loss_inner = self.compiled_loss(y, y_pred_inner)
                grad = grad_tape.gradient(loss_inner, var)

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

        avg_update_norm = tf.reduce_mean(update_norms) if update_norms else 0

        # Apply scaled gradient update to non-sampled variables
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
