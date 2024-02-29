
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





