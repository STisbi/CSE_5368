# Tisbi, Seth-Amittai
# 1000-846-338
# 2019-10-25
# Assignment-03-01

# using tensorflow_version 2.x
import tensorflow as tf
import numpy as np
import copy


class MultiNN(object):
    def __init__(self, input_dimension):
        """
        Initialize multi-layer neural network
        :param input_dimension: The number of dimensions for each the input data sample
        """
        self.input_dimension = input_dimension
        self.weights = []
        self.biases = []
        self.activations = []
        self.loss = None

    def add_layer(self, num_nodes, activation_function):
        """
         This function adds a dense layer to the neural network
         :param num_nodes: number of nodes in the layer
         :param activation_function: Activation function for the layer
         :return: None
         """

        # If this is the first layer, the dimensions are simply [input_dimensions][num_nodes]
        if not self.weights:
            # Create a RxS array of random numbers from the normal distribution
            npWeights = np.array([[np.random.normal() for column in range(num_nodes)] for row in range(self.input_dimension)])
        # Otherwise, it will be [num_nodes of previous layers][num_nodes]
        else:
            npWeights = np.array([[np.random.normal() for column in range(num_nodes)] for row in range(self.weights[-1].shape[1])])


        tfWeights = tf.Variable(npWeights, trainable=True)

        # Create a 1xS array of random numbers from the normal distribution
        npBias = np.array([[np.random.normal() for column in range(num_nodes)]])
        tfBias = tf.Variable(npBias, trainable=True)

        # Add all values into the list
        self.weights.append(tfWeights)
        self.biases.append(tfBias)
        self.activations.append(activation_function)

    def get_weights_without_biases(self, layer_number):
        """
        This function should return the weight matrix (without biases) for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
         :param layer_number: Layer number starting from layer 0.
         :return: Weight matrix for the given layer (not including the biases). Note that the shape of the weight matrix should be
          [input_dimensions][number of nodes]
         """

        return self.weights[layer_number]

    def get_biases(self, layer_number):
        """
        This function should return the biases for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
         :param layer_number: Layer number starting from layer 0
         :return: Weight matrix for the given layer (not including the biases). Note that the biases shape should be [1][number_of_nodes]
         """

        return self.biases[layer_number]

    def set_weights_without_biases(self, weights, layer_number):
        """
        This function sets the weight matrix for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
         :param weights: weight matrix (without biases). Note that the shape of the weight matrix should be
          [input_dimensions][number of nodes]
         :param layer_number: Layer number starting from layer 0
         :return: none
         """
        self.weights[layer_number] = weights

    def set_biases(self, biases, layer_number):
        """
        This function sets the biases for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
        :param biases: biases. Note that the biases shape should be [1][number_of_nodes]
        :param layer_number: Layer number starting from layer 0
        :return: none
        """

        self.biases[layer_number] = biases

    def printWeights(self):
        print("*************************************************************************************")
        for layer in range(len(self.weights)):
            print("Layer: ", str(layer + 1), "\n", "Type: ", type(self.weights[layer]), "\n", self.weights[layer])
        print("*************************************************************************************")

    def set_loss_function(self, loss_fn):
        """
        This function sets the loss function.
        :param loss_fn: Loss function
        :return: none
        """
        self.loss = loss_fn

    def sigmoid(self, x):

        return tf.nn.sigmoid(x)

    def linear(self, x):
        return x

    def relu(self, x):
        out = tf.nn.relu(x)
        return out

    def cross_entropy_loss(self, y, y_hat):
        """
        This function calculates the cross entropy loss
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
         the desired (true) class.
        :param y_hat: Array of actual outputs values [n_samples][number_of_classes].
        :return: loss
        """
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=y_hat))

    def predict(self, X):
        """
        Given array of inputs, this function calculates the output of the multi-layer network.
        :param X: Array of input [n_samples,input_dimensions].
        :return: Array of outputs [n_samples,number_of_classes ]. This array is a numerical array.
        """
        yhat = tf.Variable(X)

        for layer in range(len(self.weights)):
            # Dot product of the weights and input matrix
            weightedX = tf.matmul(yhat, self.get_weights_without_biases(layer), name="WeightedX")

            # Add the bias
            biasedX = tf.add(weightedX, self.get_biases(layer), "BiasedX")

            # Call the activation function
            yhat = self.activations[layer](biasedX)

        return yhat


    def train(self, X_train, y_train, batch_size, num_epochs, alpha=0.8, regularization_coeff=1e-6):
        """
         Given a batch of data, and the necessary hyperparameters,
         this function trains the neural network by adjusting the weights and biases of all the layers.
         :param X: Array of input [n_samples,input_dimensions]
         :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
         the desired (true) class.
         :param batch_size: number of samples in a batch
         :param num_epochs: Number of times training should be repeated over all input data
         :param alpha: Learning rate
         :param regularization_coeff: regularization coefficient
         :return: None
         """
        for epoch in range(num_epochs):
            for sample in range(0, X_train.shape[0], batch_size):
                end_row = sample + batch_size

                # There aren't enough elements left for the batch size given
                # so just use what's left
                if end_row > X_train.shape[0]:
                    end_row = X_train.shape[0]

                # Get a sample (column from X and Y) where the size of the sample is given by the batch size
                # Use it as a tf Variable
                sampleX = tf.Variable(X_train[sample : end_row,:])
                sampleY = tf.Variable(y_train[sample : end_row])

                with tf.GradientTape(persistent=True) as tape:
                    tape.watch(self.weights)
                    tape.watch(self.biases)

                    # Get the prediction, note that it is a tensorflow Variable
                    tfResults = self.predict(sampleX)

                    loss = self.cross_entropy_loss(sampleY, tfResults)

                # Update the weights and bias for each layer
                for layer in range(len(self.weights)):
                    # Derivative of the loss with respect to each layer WEIGHT
                    dl_dw = tape.gradient(loss, self.get_weights_without_biases(layer))

                    # Derivative of the loss with respect to each layer BIAS
                    dl_db = tape.gradient(loss, self.get_biases(layer))

                    # Scale the weighted derivative using alpha
                    scaled_dl_dw = tf.scalar_mul(alpha, dl_dw)

                    # Scale the weighted bias using alpha
                    scaled_dl_db = tf.scalar_mul(alpha, dl_db)

                    # Add the scaled weighted derivative to the old weights
                    new_weights = tf.subtract(self.get_weights_without_biases(layer), scaled_dl_dw)

                    # Add the scaled biased derivative to the old bias
                    new_bias = tf.subtract(self.get_biases(layer), scaled_dl_db)

                    # Update
                    self.set_weights_without_biases(new_weights, layer)
                    self.set_biases(new_bias, layer)

    def calculate_percent_error(self, X, y):
        """
        Given input samples and corresponding desired (true) output as indexes,
        this method calculates the percent error.
        For each input sample, if the predicted class output is not the same as the desired class,
        then it is considered one error. Percent error is number_of_errors/ number_of_samples.
        :param X: Array of input [n_samples,input_dimensions]
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
        the desired (true) class.
        :return percent_error
        """
        result = self.predict(X).numpy()

        # Convert the sample Y into a one hot vector or matrix
        one_hot_expected = self.toOneHot(y).transpose()

        # Find the index of the max value from each row
        max_index = result.argmax(axis=1)

        # Change the value of max value from each row to 1, and the rest to 0
        one_hot_result = (max_index[:, None] == np.arange(result.shape[1])).astype(float)

        errors = 0

        for sample in range(result.shape[0]):
            e = one_hot_expected[sample]
            x = one_hot_result[sample]
            if not np.allclose(e, x):
                errors += 1

        percent_error = errors / result.shape[0]

        return percent_error




    def calculate_confusion_matrix(self, X, y):
        """
        Given input samples and corresponding desired (true) output as indexes,
        this method calculates the confusion matrix.
        :param X: Array of input [n_samples,input_dimensions]
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
        the desired (true) class.
        :return confusion_matrix[number_of_classes,number_of_classes].
        Confusion matrix should be shown as the number of times that
        an image of class n is classified as class m where 1<=n,m<=number_of_classes.
        """
        confusion = np.zeros((self.weights[-1].shape[1], self.weights[-1].shape[1]))

        result = self.predict(X).numpy()

        # Convert the sample Y into a one hot vector or matrix
        one_hot_expected = self.toOneHot(y).transpose()

        # Find the index of the max value from each row
        max_index = result.argmax(axis=1)

        # Change the value of max value from each row to 1, and the rest to 0
        one_hot_result = (max_index[:, None] == np.arange(result.shape[1])).astype(float)

        # Each row is a sample
        for sample in range(one_hot_result.shape[0]):
            # Get what class predict thought it was
            indices = np.where(one_hot_result[sample] == 1)

            if indices[0].size != 0:
                predClass = indices[0][0]

                if not np.array_equal(one_hot_result[sample], one_hot_expected[sample]):
                    confusion[y[0], predClass] += 1
                else:
                    confusion[predClass, predClass] += 1

        return confusion


    def toOneHot(self, Y):
        oneHot = np.zeros((self.weights[-1].shape[1], Y.shape[0]))

        oneHot[Y, np.arange(Y.shape[0])] = 1

        return oneHot

if __name__ == "__main__":
    from tensorflow.keras.datasets import mnist

    np.random.seed(seed=1)
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # Reshape and Normalize data
    X_train = X_train.reshape(-1, 784).astype(np.float64) / 255.0 - 0.5
    y_train = y_train.flatten().astype(np.int32)
    input_dimension = X_train.shape[1]
    indices = list(range(X_train.shape[0]))
    # np.random.shuffle(indices)
    number_of_samples_to_use = 500
    X_train = X_train[indices[:number_of_samples_to_use]]
    y_train = y_train[indices[:number_of_samples_to_use]]
    multi_nn = MultiNN(input_dimension)
    number_of_classes = 10
    activations_list = [multi_nn.sigmoid, multi_nn.sigmoid, multi_nn.linear]
    number_of_neurons_list = [50, 20, number_of_classes]
    for layer_number in range(len(activations_list)):
        multi_nn.add_layer(number_of_neurons_list[layer_number], activation_function=activations_list[layer_number])
    for layer_number in range(len(multi_nn.weights)):
        W = multi_nn.get_weights_without_biases(layer_number)
        W = tf.Variable((np.random.randn(*W.shape)) * 0.1, trainable=True)
        multi_nn.set_weights_without_biases(W, layer_number)
        b = multi_nn.get_biases(layer_number=layer_number)
        b = tf.Variable(np.zeros(b.shape) * 0, trainable=True)
        multi_nn.set_biases(b, layer_number)
    multi_nn.set_loss_function(multi_nn.cross_entropy_loss)
    percent_error = []
    for k in range(10):
        multi_nn.train(X_train, y_train, batch_size=100, num_epochs=20, alpha=0.8)
        percent_error.append(multi_nn.calculate_percent_error(X_train, y_train))
    confusion_matrix = multi_nn.calculate_confusion_matrix(X_train, y_train)
    print("Percent error: ", np.array2string(np.array(percent_error), separator=","))
    print("************* Confusion Matrix ***************\n", np.array2string(confusion_matrix, separator=","))
