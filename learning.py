import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

from random import randint

from imageparse import process_all_images
from kmeans import cluster_and_prepare,prepare_without_clustering

default_config = {
    "image_dimensions": (155, 135),
    "block_size": 5,
    "cluster_count": 3
}

default_sources = {
    "folder_names": [],
    "image_filepaths": [],
    "text_filepaths": ["eight.txt"],
}

# Different model configurations.
model_config = [
    (1, 64),
    (2, 64),
    (3, 64),
    (1, 128),
    (2, 128),
    (3, 128),
    (1, 256),
    (2, 256),
    (3, 256)
]

prediction_sources = {
    "folder_names": ["predict-8"]
}

def load_data(config = None, sources = None):
    # Default settings are assumed if not provided.
    if config == None:
        config = default_config
    if sources == None:
        sources = default_sources
    
    image_dimensions = config["image_dimensions"]
    block_size = config["block_size"]
    cluster_count = config["cluster_count"]

    # Loads Training/Evaluation Data
    # =================================================

    folder_names = sources["folder_names"]
    image_filepaths = sources["image_filepaths"]
    text_filepaths = sources["text_filepaths"]
    
    print("Loading training/evaluation images...")
    image_matrix_list = training_matrix_list = process_all_images(folder_names, image_filepaths, text_filepaths, image_dimensions)
    
    # Clusters Images and
    # splits data into training and evaluation sublists
    # =================================================

    print("Processing and clustering training/evaluation images...")
    data = cluster_and_prepare(image_matrix_list, block_size, cluster_count)

    training_vector_count = len(data[0])
    evaluation_vector_count = len(data[3])
    test_to_total_ratio = evaluation_vector_count / (training_vector_count + evaluation_vector_count)

    print("Ready to Train and Evaluate Model.")
    print(F"Training Vectors: {training_vector_count}")
    print(F"Evaluation Vectors: {evaluation_vector_count}")
    print(F"Test/Total Ratio: {round(100 * test_to_total_ratio, 4)} %")
    input("Press Enter to Create Model. ")
    return data

def create_model(data, hidden_layers=2, nodes_per_layer=128):

    (training_vectors, training_filepaths, training_labels, evaluation_vectors, evaluation_filepaths, evaluation_labels, cluster_count) = data

    layer_list = []

    # Add a hidden layer as many times as requested.
    for i in range(hidden_layers):
        # The hidden layer is added with the specified number of neurons.
        layer_list.append(keras.layers.Dense(nodes_per_layer, activation="relu"))

    layer_list.append(keras.layers.Dense(cluster_count))

    model = keras.Sequential(layer_list)

    model.compile(optimizer="adam",
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    model.fit(training_vectors, training_labels, epochs=20)

    test_loss, test_accuracy = model.evaluate(evaluation_vectors, evaluation_labels, verbose=2)

    print("\nEvaluation Accuracy:", test_accuracy)

    return (model, data)

def make_predictions(model):
    # Loads the data.
    folder_names = prediction_sources["folder_names"]
    image_dimensions = default_config["image_dimensions"]
    block_size = default_config["block_size"]
    print("Loading data to perform predictions on...")
    prediction_data = process_all_images(folder_names, [], [], image_dimensions)
    print("Data loaded.")
    # Prepares the data for TensorFlow.
    (tf_data, filepaths) = prepare_without_clustering(prediction_data, block_size)
    input("Press Enter to Start Predicting. ")
    results = model(tf_data)
    # Initializes the class predictions list.
    class_predictions = [-1 for i in range(len(filepaths))]
    for i in range(len(filepaths)):
        # Gets the raw output from the model for the ith entry.
        output_values = results[i]
        # Finds the class index corresponding to the largest value.
        class_count = len(output_values)
        best_index = 0
        highest_value = output_values.numpy()[0]
        for class_index in range(1, class_count):
            value = output_values.numpy()[class_index]
            if value > highest_value:
                best_index = class_index
                highest_value = value
        class_predictions[i] = best_index
    return (filepaths, class_predictions)



if __name__ == "__main__":

    data = load_data()

    # Iterates over the model configurations
    for t in model_config:
        hidden_layers = t[0]
        nodes_per_layer = t[1]
        (model, _data) = create_model(data, hidden_layers, nodes_per_layer)
        print(F"Hidden Layers: {hidden_layers}")
        print(F"Nodes Per Layer: {nodes_per_layer}")
        input("Press Enter to Create the Next Model. ")
