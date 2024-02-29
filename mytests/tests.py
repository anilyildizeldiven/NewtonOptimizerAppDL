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

# Import necessary libraries for testing, numerical operations, TensorFlow, and data preprocessing.
import unittest
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Define a test case class for the NewtonOptimizedModel by inheriting from unittest.TestCase.
class TestNewtonOptimizedModel(unittest.TestCase):

    def setUp(self):
        # This method is called before each test. It prepares the data and model for testing.
        np.random.seed(42)  # Set a random seed for reproducibility.
        self.X = np.random.rand(100, 4)  # Generate synthetic feature data: 100 samples, 4 features each.
        self.y = np.random.randint(0, 3, 100)  # Generate synthetic labels: 100 samples, 3 possible classes.
        encoder = LabelEncoder()
        self.y_encoded = to_categorical(encoder.fit_transform(self.y))  # Encode labels into one-hot vectors.

        # Split the data into training and testing sets.
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y_encoded, test_size=0.2, random_state=42)
        
        # Create and compile the model.
        self.model = NewtonOptimizedModel()  # Instantiate the model.
        self.model.compile(loss='categorical_crossentropy', metrics=['accuracy'])  # Compile the model with a loss function and accuracy metric.

    def test_model_training(self):
        # Test whether the model can be trained on the synthetic data.
        batch_size = self.X_train.shape[0]  # Use the entire training set as the batch size.

        # Train the model for one epoch.
        history = self.model.fit(self.X_train, self.y_train, batch_size=batch_size, epochs=1, validation_split=0.25)

        # Check if the training process has affected the 'loss' and 'accuracy' metrics.
        self.assertIn('loss', history.history)  # Verify 'loss' is in the training history.
        self.assertIn('accuracy', history.history)  # Verify 'accuracy' is in the training history.

        # Check if the accuracy is reasonable (e.g., better than random guessing).
        accuracy = history.history['accuracy'][0]
        self.assertGreater(accuracy, 1/5)  # Assuming random guessing would have a probability of 1/3, expect better performance.

    def test_model_weights_change_after_training(self):
        # Test if the model's weights change after training.
        batch_size = self.X_train.shape[0]
    
        # Save the initial weights of the model.
        initial_weights = [layer.get_weights() for layer in self.model.layers]
    
        # Train the model on synthetic data.
        self.model.fit(self.X_train, self.y_train, batch_size=batch_size, epochs=1, validation_split=0.25)
    
        # Save the updated weights after training.
        updated_weights = [layer.get_weights() for layer in self.model.layers]
    
        # Check if at least one weight has changed.
        for initial_layer_weights, updated_layer_weights in zip(initial_weights, updated_weights):
            for initial_weight, updated_weight in zip(initial_layer_weights, updated_layer_weights):
                # Assert that the weights have changed by comparing the initial and updated weights.
                self.assertFalse(np.array_equal(initial_weight, updated_weight), "A weight did not change as expected.")

    def test_model_prediction(self):
        # Test the model's prediction capability.
        random_input = np.random.rand(1, 4)  # Generate a random input vector.
        predictions = self.model.predict(random_input)  # Make a prediction.
    
        # Check if the prediction output has the correct shape.
        self.assertEqual(predictions.shape, (1, 3))  # Assuming there are 3 classes.
    
        # Verify that the predictions are valid probabilities.
        self.assertTrue(np.all(predictions >= 0) and np.all(predictions <= 1))  # Probabilities must be between 0 and 1.
        self.assertAlmostEqual(np.sum(predictions), 1.0, places=5)  # Sum of probabilities should be 1.
        
    def test_model_initialization(self):
        # Test the initial configuration of the model.
        expected_number_of_layers = 3  # Adjust based on your model's configuration.
        self.assertEqual(len(self.model.layers), expected_number_of_layers)  # Check the number of layers.
    
        # Check the activation function of the output layer.
        output_activation = self.model.output_layer.activation.__name__
        self.assertEqual(output_activation, 'softmax')  # The output layer should use softmax for multi-class classification.

    def test_training_with_various_batch_sizes(self):
        # Test the model's ability to train with various batch sizes.
        for batch_size in [1, 10, len(self.X_train)]:
            # Attempt to train the model with different batch sizes.
            try:
                self.model.fit(self.X_train, self.y_train, batch_size=batch_size, epochs=1, validation_split=0.25)
            except Exception as e:
                # If training fails, the test should fail with an informative message.
                self.fail(f"Training failed with batch size {batch_size}: {e}")

    def test_hessian_regularization(self):
        """Test the Hessian matrix regularization."""
        # Generate a sample Hessian matrix and apply regularization.
        sample_hessian = tf.random.uniform((4, 4), dtype=tf.float32)
        regularized_hessian = self.model.regularize_hessian(sample_hessian, regularization_strength=1)

        # Check if the diagonal elements have been properly regularized.
        for i in range(4):
            self.assertTrue(regularized_hessian[i][i] > sample_hessian[i][i].numpy(), "Hessian matrix regularization failed.")


# This block allows the test suite to be run from the command line.
if __name__ == '__main__':
    unittest.main()

    



def simulate_training(model, epochs, subsampling_parameter, data):
    """Simuliert den Trainingsprozess für eine gegebene Anzahl von Epochen."""
    history = {'loss': [], 'accuracy': []}
    for _ in range(epochs):
        result = model.train_step(data, subsampling_parameter=subsampling_parameter)
        history['loss'].append(result['loss'])
        history['accuracy'].append(result['accuracy'])
    return history

class TestSubsamplingOverTime(unittest.TestCase):
    def setUp(self):
        self.model = NewtonOptimizedModel()
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.x_train = np.random.random((100, 4)).astype(np.float32)
        self.y_train = tf.keras.utils.to_categorical(np.random.randint(3, size=(100, 1)), num_classes=3)
        self.data = (tf.convert_to_tensor(self.x_train), tf.convert_to_tensor(self.y_train))

    def test_subsampling_effect_over_time(self):
        # Simulieren Sie das Training mit und ohne Subsampling
        history_with_subsampling = simulate_training(self.model, epochs=10, subsampling_parameter=1, data=self.data)
        history_without_subsampling = simulate_training(self.model, epochs=10, subsampling_parameter=0, data=self.data)

        # Analysieren Sie die Trends in Verlust- und Genauigkeitswerten
        # Zum Beispiel: Überprüfen Sie, ob sich die Endwerte signifikant unterscheiden
        self.assertNotEqual(history_with_subsampling['loss'][-1], history_without_subsampling['loss'][-1], "Subsampling hat keinen langfristigen Einfluss.")
        # Weitere Analysen könnten hier folgen, z.B. Trendvergleiche, statistische Tests etc.

if __name__ == '__main__':
    unittest.main()





