import tensorflow as tf 
from tensorflow.keras import layers
import numpy as np
import conf
import ocp_double_pendulum as ocp_double
import pandas as pd
from sklearn.model_selection import train_test_split

def create_model(input_shape):
    inputs = layers.Input(shape=(input_shape,))
    out1 = layers.Dense(64, activation='relu')(inputs)
    out2 = layers.Dense(32, activation='relu')(out1)
    out3 = layers.Dense(16, activation='relu')(out2)
    out4 = layers.Dense(4, activation='relu')(out3)
    outputs = layers.Dense(1, activation='relu')(out4)

    model = tf.keras.Model(inputs, outputs)
    return model

def rerun(model):
    _ , state_array = conf.random_states(10000)

    # Make predictions on the test set
    to_test = conf.scaler.fit_transform(state_array)
    label_pred = model.predict(to_test)

    to_check_again = []

    for i, label in enumerate(label_pred):
        if (label > 0 and label < 1):
            to_check_again.append(state_array[i,:])

    newdf = ocp_double.check(to_check_again)

    return newdf

## Dataset creation

if __name__ == "__main__":
    # Import dataset and labels from configuration file
    train_data = conf.train_data
    train_label = conf.train_label
    test_data = conf.test_data
    test_label = conf.test_label

    model = create_model(4)
    model.load_weights("iterata2.h5")

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(train_data, train_label, epochs=500)
    
    # Test trained neural network
    results = model.evaluate(test_data, test_label)
    print("Test accuracy:", results[1])

    for i in range(conf.n_iterations):
        newdf = rerun(model)
        labels = newdf['viable']
        dataset = newdf.drop('viable', axis=1)
        train_size = 0.8
        train_data, test_data, train_label, test_label = train_test_split(dataset, labels, train_size=train_size, random_state=17)
        train_data = conf.scaler.fit_transform(train_data)
        test_data = conf.scaler.transform(test_data)
        print("Active learning run", i+1, "out of", conf.n_iterations)
        model.fit(train_data, train_label, epochs=500)
        results = model.evaluate(test_data, test_label)
        print("Test accuracy:", results[1])

    # Save the model weights
    model.save_weights("iterata3.h5")
