import unittest
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from subsampledNewton import NewtonOptimizedModel  # Stellen Sie sicher, dass Sie den korrekten Importpfad verwenden

class TestNewtonOptimizedModel(unittest.TestCase):

    def setUp(self):
        # Synthetische Daten für das Testen
        np.random.seed(42)  # Für reproduzierbare Ergebnisse
        self.X = np.random.rand(100, 4)  # 100 Beispiele, 4 Features
        self.y = np.random.randint(0, 3, 100)  # 100 Beispiele, 3 Klassen
        encoder = LabelEncoder()
        self.y_encoded = to_categorical(encoder.fit_transform(self.y))

        # Daten aufteilen
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y_encoded, test_size=0.2, random_state=42)
        
        # Modell erstellen und kompilieren
        self.model = NewtonOptimizedModel()
        self.model.compile(loss='categorical_crossentropy', metrics=['accuracy'])

    def test_model_training(self):
        # Training Parameters
        batch_size = self.X_train.shape[0]

        # Train the model on synthetic data
        history = self.model.fit(self.X_train, self.y_train, batch_size=batch_size, epochs=1, validation_split=0.25)

        # Überprüfen, ob das Training erfolgreich war
        self.assertIn('loss', history.history)
        self.assertIn('accuracy', history.history)

        # Überprüfen, ob die Genauigkeit sinnvoll ist (z.B. besser als Raten)
        accuracy = history.history['accuracy'][0]
        self.assertGreater(accuracy, 1/5)  # Angenommen, zufälliges Raten hätte eine Wahrscheinlichkeit von 1/3

    def test_model_weights_change_after_training(self):
        # Training Parameters
        batch_size = self.X_train.shape[0]
    
        # Gewichte vor dem Training speichern
        initial_weights = [layer.get_weights() for layer in self.model.layers]
    
        # Modell auf synthetischen Daten trainieren
        self.model.fit(self.X_train, self.y_train, batch_size=batch_size, epochs=1, validation_split=0.25)
    
        # Gewichte nach dem Training speichern
        updated_weights = [layer.get_weights() for layer in self.model.layers]
    
        # Überprüfen, ob sich mindestens ein Gewicht geändert hat
        for initial_layer_weights, updated_layer_weights in zip(initial_weights, updated_weights):
            for initial_weight, updated_weight in zip(initial_layer_weights, updated_layer_weights):
                # Überprüfen, ob Gewichte gleich sind; np.array_equal gibt False zurück, wenn sie sich unterscheiden
                self.assertFalse(np.array_equal(initial_weight, updated_weight), "Ein Gewicht hat sich nicht wie erwartet verändert.")
    
    def test_model_prediction(self):
        # Zufällige Daten für die Vorhersage generieren
        random_input = np.random.rand(1, 4)
        predictions = self.model.predict(random_input)
    
        # Überprüfen, ob die Vorhersage die richtige Form hat
        self.assertEqual(predictions.shape, (1, 3))  # Angenommen, es gibt 3 Klassen
    
        # Überprüfen, ob die Vorhersagen gültige Wahrscheinlichkeiten sind
        self.assertTrue(np.all(predictions >= 0) and np.all(predictions <= 1))
        self.assertAlmostEqual(np.sum(predictions), 1.0, places=5)
        
    def test_model_initialization(self):
        # Überprüfen der Anzahl der Schichten
        expected_number_of_layers = 3  # Anpassen basierend auf Ihrem Modell
        self.assertEqual(len(self.model.layers), expected_number_of_layers)
    
        # Überprüfen der Aktivierungsfunktion der Ausgabeschicht
        output_activation = self.model.output_layer.activation.__name__
        self.assertEqual(output_activation, 'softmax')

    def test_training_with_various_batch_sizes(self):
        for batch_size in [1, 10, len(self.X_train)]:
            # Versuchen, mit unterschiedlichen Batch-Größen zu trainieren
            try:
                self.model.fit(self.X_train, self.y_train, batch_size=batch_size, epochs=1, validation_split=0.25)
            except Exception as e:
                self.fail(f"Training failed with batch size {batch_size}: {e}")

    def test_hessian_regularization(self):
        """Testet die Regularisierung der Hesse-Matrix."""
        # Generieren Sie eine beispielhafte Hesse-Matrix und wenden Sie die Regularisierung an
        sample_hessian = tf.random.uniform((4, 4), dtype=tf.float32)
        regularized_hessian = self.model.regularize_hessian(sample_hessian,regularization_strength=1)

        # Überprüfen Sie, ob die Diagonalelemente wie erwartet regularisiert wurden
        for i in range(4):
            self.assertTrue(regularized_hessian[i][i] > sample_hessian[i][i].numpy(), "Regularisierung der Hesse-Matrix fehlgeschlagen.")

if __name__ == '__main__':
    unittest.main()
    
    









