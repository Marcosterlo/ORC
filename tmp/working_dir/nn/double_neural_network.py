import tensorflow as tf 
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import working_dir.double_pendulum.mpc_double_pendulum_conf as conf

def create_model(input_shape):
    inputs = layers.Input(shape=(input_shape,))
    out1 = layers.Dense(64, activation='relu')(inputs)
    out2 = layers.Dense(32, activation='relu')(out1)
    out3 = layers.Dense(16, activation='relu')(out2)
    out4 = layers.Dense(4, activation='relu')(out3)
    outputs = layers.Dense(1, activation='relu')(out4)

    model = tf.keras.Model(inputs, outputs)
    return model

## Dataset creation

if __name__ == "__main__":
    # Import dataset and labels from configuration file
    train_data = conf.train_data
    train_label = conf.train_label
    test_data = conf.test_data
    test_label = conf.test_label

    model = create_model(4)
    print(model.summary())

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(train_data, train_label, epochs=100)

    # Save the model weights
    model.save_weights("double_pendulum_prova3.h5")

    # Test trained neural network
    results = model.evaluate(test_data, test_label)
    print("Test accuracy:", results[1])

    # viable_states = []
    # no_viable_states = []

    # Creation of initial states grid
    # _ , state_array = conf.grid_states(50, 50)
    # _ , state_array = conf.random_states(4000)

    # Make predictions on the test set
    # to_test = conf.scaler.fit_transform(state_array)
    # label_pred = model.predict(to_test)
