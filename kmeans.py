from numpy import array
from random import choice,randint

def euclidean_distance_squared(vector1, vector2):
    '''Returns the square of the Euclidean distance between two vectors.'''
    # Finds the difference between these vectors.
    difference = vector1 - vector2
    # Squares each component of the difference vector.
    squared_difference = difference ** 2
    # Takes the sum of the squared components.
    return sum(squared_difference)

def calc_cluster_mean(cluster, vector_dimension):
    cluster_size = len(cluster)
    # If this is an empty cluster...
    if cluster_size == 0:
        # ...set the mean to a vector filled with NaN values.
        return array([float("nan") for i in range(vector_dimension)])
    # Initializes the total to a zero vector.
    total = array([0.0 for i in range(vector_dimension)])
    # Iterates over the vectors in the cluster.
    for vector in cluster:
        # Adds the vector to the total.
        total += vector
    return total / cluster_size

class ImageVector:
    def __init__(self, matrix, block_size):
        self.filepath = matrix.filepath
        self.data_array = array(matrix.to_vector(block_size))

def k_means_clustering(k, data_list):
    # Gets the number of vectors.
    object_count = len(data_list)
    if k > object_count:
        raise ValueError(F'Cluster count (k == {k}) exceeds number of vectors ({object_count}).')
    # Gets the dimension of each vector.
    dimension = len(data_list[0].data_array)
    # This list will hold the indices of the initial means.
    mean_indices = []
    # Iterates until k means have been selected.
    while len(mean_indices) < k:
        # Selects a random index.
        random_index = choice(range(object_count))
        # Checks if the index has not been selected yet.
        if random_index not in mean_indices:
            mean_indices.append(random_index)
    # Gets the values at the selected indices.
    cluster_means = [data_list[i].data_array for i in mean_indices]
    print(F'\nIteration 0: Cluster Means:\n{array(cluster_means)}\n')
    # Initializes the cluster assignments.
    old_cluster_assignments = None
    new_cluster_assignments = [None for i in range(object_count)]
    iteration_count = 1
    # Iterates until the clusters converge.
    while old_cluster_assignments != new_cluster_assignments:
        # Moves the cluster assignments from the previous iteration to old_cluster_assignments.
        old_cluster_assignments = new_cluster_assignments
        # Initializes the new cluster assignments list.
        new_cluster_assignments = [-1 for i in range(object_count)]
        # Iterates over the vector indices.
        for vector_index in range(object_count):
            vector = data_list[vector_index].data_array
            closest_mean_index = None
            distance_to_closest_mean = float("inf")
            # Iterates over the cluster indices.
            for cluster_index in range(k):
                # Gets the mean of the cluster.
                mean_value = cluster_means[cluster_index]
                # Calculates the distance to this cluster mean.
                distance_to_mean = euclidean_distance_squared(vector, mean_value)
                # Tests if this distance is less than the current min.
                if distance_to_mean < distance_to_closest_mean:
                    # Sets this mean as the new closest mean.
                    closest_mean_index = cluster_index
                    distance_to_closest_mean = distance_to_mean
            # The vector is assigned to the cluster with the closest mean.
            new_cluster_assignments[vector_index] = closest_mean_index
        # Computes the new mean of each cluster.
        for cluster_index in range(k):
            # Filters the data list to only include vectors in this cluster.
            cluster = [data_list[i].data_array for i in range(object_count) if new_cluster_assignments[i] == cluster_index]
            # Calculates the mean of the vectors in this cluster.
            cluster_means[cluster_index] = calc_cluster_mean(cluster, dimension)
        print(F'\nIteration {iteration_count}: Cluster Means:\n{array(cluster_means)}\n')
        iteration_count += 1
    print("Clustering Complete. Results:")
    # This list will hold the size of every cluster.
    cluster_sizes = [0 for i in range(k)]
    for vector_index in range(object_count):
        # Finds the cluster that the vector at this index belongs to.
        cluster_index = new_cluster_assignments[vector_index]
        # Increments the size of whatever cluster this vector belongs to.
        cluster_sizes[cluster_index] += 1
    for cluster_index in range(k):
        print(F"Cluster {cluster_index} Size: {cluster_sizes[cluster_index]}")
    return new_cluster_assignments

def make_data_list(image_matrix_list, block_size):
    return [ImageVector(matrix, block_size) for matrix in image_matrix_list.matrices]

def make_shuffled_list(size, iterations=None):
    '''Generates a list of integers from 0 to (size - 1) in random order.'''
    print("Creating shuffled list...")
    # Creates the initial (unshuffled) list.
    new_list = [i for i in range(size)]
    # The number of iterations defaults to the size of the list.
    if iterations == None:
        iterations = size
    for i in range(iterations):
        # Moves the previous list into the old_list variable.
        old_list = new_list
        # Creates an empty list to fill with shuffled integers.
        new_list = []
        # Iterate until the old list is empty.
        while len(old_list) > 0:
            # Randomly decides the index of the number to remove from old_list.
            index = randint(0, len(old_list) - 1)
            # Removes the element from old_list.
            element = old_list.pop(index)
            # Appends the element to new_list.
            new_list.append(element)
    print("Finished creating shuffled list.")
    return new_list

def split_data(data_list, object_labels):
    '''Splits data and labels into training and evaluation categories.'''
    # Elements will be selected from the data/object list in a random order
    # determined by this list of shuffled indices.
    indices = make_shuffled_list(len(data_list))
    # Initializes the new lists.
    training_data = []
    training_labels = []
    evaluation_data = []
    evaluation_labels = []
    for i in range(len(data_list)):
        # Selects a vector and its corresponding label.
        vector = data_list[i]
        label = object_labels[i]
        # If i is not divisible by 5, the data and label are used for training.
        if i % 5 != 0:
            training_data.append(vector)
            training_labels.append(label)
        # Otherwise, the data and label are used for evaluation.
        else:
            evaluation_data.append(vector)
            evaluation_labels.append(label)
    return (training_data, training_labels, evaluation_data, evaluation_labels)

def prepare_data_list_for_tf(data_list):
    '''Prepares data list to be processed with TensorFlow.'''
    # Removes the metadata from the vectors, wraps the vectors in
    # numpy arrays, and wraps the whole list in a numpy array.
    vector_count = len(data_list)
    tf_data = array([array(data_list[i].data_array) for i in range(vector_count)])
    # The filepaths are placed in a separate list.
    filepaths = [data_list[i].filepath for i in range(vector_count)]
    return (tf_data, filepaths)

def cluster_and_prepare(image_matrix_list, block_size, cluster_count):
    # Creates the data list from the ImageMatrixList.
    data_list = make_data_list(image_matrix_list, block_size)
    # Clusters the data.
    object_labels = k_means_clustering(cluster_count, data_list)
    
    # Splits the data and labels into training and evaluation categories.
    (training_data, training_labels, evaluation_data, evaluation_labels) = split_data(data_list, object_labels)
    # Prepares all four lists for TensorFlow.
    (tf_training_data, training_filepaths) = prepare_data_list_for_tf(training_data)
    tf_training_labels = array(training_labels)
    (tf_evaluation_data, evaluation_filepaths) = prepare_data_list_for_tf(evaluation_data)
    tf_evaluation_labels = array(evaluation_labels)

    return (tf_training_data, training_filepaths, tf_training_labels, tf_evaluation_data, evaluation_filepaths, tf_evaluation_labels, cluster_count)

def prepare_without_clustering(image_matrix_list, block_size):
    # Creates the data list from the ImageMatrixList.
    data_list = make_data_list(image_matrix_list, block_size)
    # Prepares the list for TensorFlow.
    (data, filepaths) = prepare_data_list_for_tf(data_list)
    return (data, filepaths)