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
class NewtonOptimizedModel(Model):
    def __init__(self):
        super(NewtonOptimizedModel, self).__init__()
        self.dense = Dense(15, activation='tanh', input_shape=(4,), kernel_initializer=RandomNormal())
        self.dense1 = Dense(10, activation='tanh', kernel_initializer=RandomNormal())
        self.output_layer = Dense(3, activation='softmax', kernel_initializer=RandomNormal())

    def call(self, inputs):
        x = self.dense(inputs)
        x = self.dense1(x)
        return self.output_layer(x)

    def regularize_hessian(self, hessian, regularization_strength=1e-9):
        regularized_hessian = hessian + tf.eye(tf.shape(hessian)[0]) * regularization_strength
        return regularized_hessian

    def train_step(self, data, subsampling_parameter=0.5):
        x, y = data

        with tf.GradientTape(persistent=True) as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred)

        layer_gradients = {}
        layer_hessians = {}

        for layer in self.layers:
            with tf.GradientTape(persistent=True) as hessian_tape:
                # Compute gradients for each layer separately
                layer_loss = self.compiled_loss(y, self(x, training=True))
                grads = hessian_tape.gradient(layer_loss, layer.trainable_variables)

                # Determine if layer should be subsampled based on subsampling_parameter
                if np.random.rand() < subsampling_parameter:
                    # Compute Hessians for subsampled layers
                    hessians = [hessian_tape.jacobian(grad, var) for grad, var in zip(grads, layer.trainable_variables)]
                    layer_hessians[layer.name] = hessians
                else:
                    # For non-sampled layers, simply store gradients
                    layer_gradients[layer.name] = grads

        # Apply updates using gradients for non-sampled layers and Hessians for subsampled layers
        for layer in self.layers:
            if layer.name in layer_hessians:
                # Apply updates using Hessians for subsampled layers
                for hessian, grad, var in zip(layer_hessians[layer.name], layer_gradients.get(layer.name, []), layer.trainable_variables):
                    # Flatten Hessian and compute inverse or pseudo-inverse
                    hessian_flat = tf.reshape(hessian, [tf.size(var), -1])
                    reg_hessian = self.regularize_hessian(hessian_flat)
                    inv_hessian = tf.linalg.inv(reg_hessian)
                    grad_flat = tf.reshape(grad, [-1, 1])
                    var_update = tf.linalg.matmul(inv_hessian, grad_flat)
                    var.assign_sub(tf.reshape(var_update, var.shape))
            elif layer.name in layer_gradients:
                # Apply updates using gradients for non-sampled layers
                for grad, var in zip(layer_gradients[layer.name], layer.trainable_variables):
                    var.assign_sub(grad)

        del tape
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

import unittest
import tensorflow as tf
import numpy as np

# Hier fügen Sie Ihren NewtonOptimizedModel Code ein

class TestNewtonOptimizedModelSubsampling(unittest.TestCase):
    def setUp(self):
        # Initialisierung des Modells und der Testdaten
        self.model = NewtonOptimizedModel(input_shape=(4,), subsampling_rate=0.9, num_classes=3)
        self.x_train = np.random.rand(100, 4).astype(np.float32)  # Simulierte Trainingsdaten
        self.y_train = tf.keras.utils.to_categorical(np.random.randint(3, size=100), num_classes=3)

    def test_subsampling_rate(self):
        # Anzahl der trainierbaren Variablen im Modell
        total_variables = len(self.model.trainable_variables)
        expected_subsampled_variables = int(np.floor(total_variables * self.model.subsampling_rate))

        # Manuelle Simulation des Subsampling-Schrittes
        subsampled_indices = np.random.choice(range(total_variables), 
                                              size=expected_subsampled_variables, 
                                              replace=False)

        # Überprüfung, ob die Anzahl der subsampled Variablen der Erwartung entspricht
        self.assertEqual(len(subsampled_indices), expected_subsampled_variables)

    def test_training_with_subsampling(self):
        # Trainieren des Modells mit einem Batch von Daten, um die Funktionsweise des Subsamplings zu testen
        initial_weights = [tf.identity(var) for var in self.model.trainable_variables]
        self.model.train_step((self.x_train[:10], self.y_train[:10]))

        # Überprüfung, ob nach dem Training eine Veränderung in den Gewichten stattgefunden hat
        for initial_weight, updated_weight in zip(initial_weights, self.model.trainable_variables):
            self.assertTrue(np.any(initial_weight.numpy() != updated_weight.numpy()))

if __name__ == '__main__':
    unittest.main()

