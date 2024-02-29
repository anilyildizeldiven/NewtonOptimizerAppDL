from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import RandomNormal

# Load Data direct
iris = load_iris()
X = iris.data
y = iris.target

# Prepare Data
encoder = LabelEncoder()
encoder.fit(y)
encoded_Y = encoder.transform(y)
dummy_y = to_categorical(encoded_Y)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, dummy_y, test_size=0.2, random_state=42)

# Create and Compile Model
model = NewtonOptimizedModel()
model.compile(loss='categorical_crossentropy', metrics=['accuracy'])

# Training Parameters
batch_size = X_train.shape[0]

# Train Model
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=300, validation_split=0.25)
