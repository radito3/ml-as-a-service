import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras import regularizers


class NeuralNetwork:
    num_layers = 4
    num_neurons_per_layer = 8

    # num_classes = len(np.unique(y))
    def __init__(self, num_classes):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(8, activation='relu', kernel_regularizer=regularizers.l2(0.01), input_shape=(4,)),
            tf.keras.layers.Dense(8, activation='relu', kernel_regularizer=regularizers.l2(0.01), input_shape=(4,)),
            tf.keras.layers.Dense(8, activation='relu', kernel_regularizer=regularizers.l2(0.01), input_shape=(4,)),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        # Compile the model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model = model

    @staticmethod
    def split_data(x, y, test_size=0.2):
        return train_test_split(x, y, test_size=test_size, random_state=42)

    @staticmethod
    def prepare_data(x_train, x_test, y_train, y_test, num_classes):
        # Apply feature scaling using StandardScaler
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)
        # Convert the labels to one-hot encoded format
        y_train = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)
        return x_train_scaled, x_test_scaled, y_train, y_test

    def fit_model(self, x, y, epochs=100):
        self.model.fit(x, y, epochs=epochs, verbose=0)

    def evaluate(self, x, y):
        loss, accuracy = self.model.evaluate(x, y)
        return accuracy, loss

    def predict(self, x):
        return np.argmax(self.model.predict(x), axis=1)

