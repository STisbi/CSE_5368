# Tisbi, Seth-Amittai
# 1000-846-338
# 2019-10-06
# Assignment-02-01

import numpy as np
import itertools
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf


def display_images(images):
    # This function displays images on a grid.
    # Farhad Kamangar Sept. 2019
    number_of_images=images.shape[0]
    number_of_rows_for_subplot=int(np.sqrt(number_of_images))
    number_of_columns_for_subplot=int(np.ceil(number_of_images/number_of_rows_for_subplot))
    for k in range(number_of_images):
        plt.subplot(number_of_rows_for_subplot,number_of_columns_for_subplot,k+1)
        plt.imshow(images[k], cmap=plt.get_cmap('gray'))
        # plt.imshow(images[k], cmap=pyplot.get_cmap('gray'))
    plt.show()

def display_numpy_array_as_table(input_array):
    # This function displays a 1d or 2d numpy array (matrix).
    # Farhad Kamangar Sept. 2019
    if input_array.ndim==1:
        num_of_columns,=input_array.shape
        temp_matrix=input_array.reshape((1, num_of_columns))
    elif input_array.ndim>2:
        print("Input matrix dimension is greater than 2. Can not display as table")
        return
    else:
        temp_matrix=input_array
    number_of_rows,num_of_columns = temp_matrix.shape
    plt.figure()
    tb = plt.table(cellText=np.round(temp_matrix,2), loc=(0,0), cellLoc='center')
    for cell in tb.properties()['child_artists']:
        cell.set_height(1/number_of_rows)
        cell.set_width(1/num_of_columns)

    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()
class Hebbian(object):
    def __init__(self, input_dimensions=2,number_of_classes=4,transfer_function="Hard_limit",seed=None):
        """
        Initialize Perceptron model
        :param input_dimensions: The number of features of the input data, for example (height, weight) would be two features.
        :param number_of_classes: The number of classes.
        :param transfer_function: Transfer function for each neuron. Possible values are:
        "Hard_limit" ,  "Sigmoid", "Linear".
        :param seed: Random number generator seed.
        """
        if seed != None:
            np.random.seed(seed)
        self.input_dimensions = input_dimensions
        self.number_of_classes=number_of_classes
        self.transfer_function=transfer_function
        self._initialize_weights()

    def _initialize_weights(self):
        # Create a Sx(R+1) list of random numbers from the normal distribution. The plus one in column 1 is for the bias
        weights = [[np.random.normal() for column in range(self.input_dimensions + 1)] for row in
                   range(self.number_of_classes)]

        # Create a numpy array with this list
        self.weights = np.array(weights)

    def initialize_all_weights_to_zeros(self):
        # Create a Sx(R+1) numpy array filled with zero's
        self.weights = np.zeros((self.number_of_classes, self.input_dimensions + 1))

    def predict(self, X):
        # Add this list (made into a numpy array) as the first row of the input matrix
        newX = np.vstack((np.array([1 for column in range(X.shape[1])]), X))

        # Multiply the weight matrix, W, by the input matrix X
        results = np.dot(self.weights, newX)

        if self.transfer_function == "Hard_limit":
            actualResults = np.where(results < 0, 0, 1)
        elif self.transfer_function == "Linear":
            actualResults = results
        elif self.transfer_function == "Sigmoid":
            actualResults = 1 / (1 + np.exp(-results))
            # actualResults[np.where(actualResults == np.max(actualResults))] = 1
            # actualResults[np.where(actualResults != 1)] = 0

        return actualResults

    def print_weights(self):
        print(self.weights)

    def train(self, X, y, batch_size=1,num_epochs=10,  alpha=0.1,gamma=0.9,learning="Delta"):
        """
        Given a batch of data, and the necessary hyperparameters,
        this function adjusts the self.weights using Perceptron learning rule.
        Training should be repeted num_epochs time.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
        the desired (true) class.
        :param batch_size: number of samples in a batch
        :param num_epochs: Number of times training should be repeated over all input data
        :param alpha: Learning rate
        :param gamma: Controls the decay
        :param learning: Learning rule. Possible methods are: "Filtered", "Delta", "Unsupervised_hebb"
        :return: None
        """

        # Iterate through an epoch then by sample
        for epoch in range(num_epochs):
            for sample in range(0, X.shape[1], batch_size):
                end_column = sample + batch_size

                # There aren't enough elements left for the batch size given
                # so just use what's left
                if end_column > X.shape[1]:
                    end_column = X.shape[1]

                # Get a sample (column from X and Y) where the size of the sample is given by the batch size
                sampleX = X[:, sample : end_column]
                sampleY = y[sample : end_column]

                # Convert the sample Y into a one hot vector or matrix
                oneHot = self.toOneHot(sampleY)

                # Get the prediction
                results = self.predict(sampleX)

                if learning == "Delta":
                    # Calculate e
                    e = np.subtract(oneHot, results)

                    # Add a row of one's to the top of the input matrix
                    newX = np.vstack((np.array([1 for column in range(sampleX.shape[1])]), sampleX))

                    # Calculate e dot p, where p is the input matrix
                    ep = np.dot(e, np.transpose(newX))

                    # Multiply this new matrix by the scalar alpha
                    rate = np.multiply(alpha, ep)

                    # Calculate the new weights along with the bias
                    self.weights = np.add(self.weights, rate)

                elif learning == "Filtered":
                    # Add a row of one's to the top of the input matrix
                    newX = np.vstack((np.array([1 for column in range(sampleX.shape[1])]), sampleX))

                    # Calculate e dot p, where p is the input matrix
                    ep = np.dot(oneHot, np.transpose(newX))

                    # Multiply this new matrix by the scalar alpha
                    rate = np.multiply(alpha, ep)

                    # Multiply the old weights by some scalar gamma
                    oldWeightMod = np.multiply(1 - gamma, self.weights)

                    self.weights = np.add(oldWeightMod, rate)

                elif learning == "Unsupervised_hebb":
                    # Add a row of one's to the top of the input matrix
                    newX = np.vstack((np.array([1 for column in range(sampleX.shape[1])]), sampleX))

                    # Calculate e dot p, where p is the input matrix
                    ep = np.dot(results, np.transpose(newX))

                    # Multiply this new matrix by the scalar alpha
                    rate = np.multiply(alpha, ep)

                    # Calculate the new weights along with the bias
                    self.weights = np.add(self.weights, rate)

    def calculate_percent_error(self, X, y):
        """
        Given a batch of data this function calculates percent error.
        For each input sample, if the predicted class output is not the same as the desired class,
        then it is considered one error. Percent error is number_of_errors/ number_of_samples.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
        the desired (true) class.
        :return percent_error
        """
        numErrors = 0

        for sample in range(X.shape[1]):
            # Get the nth column from both the input and expected output
            sampleX = np.transpose(np.array([X[:, sample]]))
            sampleY = np.transpose(np.array([y[sample]]))

            # Convert the sample Y into a one hot vector or matrix
            oneHot = self.toOneHot(sampleY)

            results = self.predict(sampleX)

            if self.transfer_function == "Hard_limit":
                pass
            elif self.transfer_function == "Linear":
                # print("results", results)

                results[np.where(results == np.max(results))] = 1
                results[np.where(results != 1)] = 0

                # print("predict: ", results)
            elif self.transfer_function == "Sigmoid":
                print("sigmoid", results)
                results[np.where(results == np.max(results))] = 1
                results[np.where(results != 1)] = 0

            if results[sampleY, 0] != 1:
                numErrors += 1

        print("Number of Errors: %d\nNumber of samples: %d"%(numErrors, X.shape[1]))

        percentError = numErrors / X.shape[1]

        print("Percent Error: ", percentError)

        return percentError

    def calculate_confusion_matrix(self,X,y):
        """
        Given a desired (true) output as one hot and the predicted output as one-hot,
        this method calculates the confusion matrix.
        If the predicted class output is not the same as the desired output,
        then it is considered one error.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
        the desired (true) class.
        :return confusion_matrix[number_of_classes,number_of_classes].
        Confusion matrix should be shown as the number of times that
        an image of class n is classified as class m where 1<=n,m<=number_of_classes.
        """
        confusion = np.zeros((self.number_of_classes, self.number_of_classes))

        for sample in range(X.shape[1]):
            # Get the nth column from both the input and expected output
            sampleX = np.transpose(np.array([X[:, sample]]))
            sampleY = np.transpose(np.array([y[sample]]))

            # Convert the sample Y into a one hot vector or matrix
            oneHot = self.toOneHot(sampleY)

            results = self.predict(sampleX)

            if self.transfer_function == "Hard_limit":
                pass
            elif self.transfer_function == "Linear":
                results[np.where(results == np.max(results))] = 1
                results[np.where(results != 1)] = 0
            elif self.transfer_function == "Sigmoid":
                pass

            # Get what class predict thought it was
            indices = np.where(results == 1)

            if indices[0].size != 0:
                predClass = indices[0][0]

                if not np.array_equal(results, oneHot):
                    confusion[sampleY[0], predClass] += 1
                else:
                    confusion[predClass, predClass] += 1

        return confusion




    def toOneHot(self, Y):
        oneHot = np.zeros((self.number_of_classes, Y.shape[0]))

        oneHot[Y, np.arange(Y.shape[0])] = 1

        return oneHot



if __name__ == "__main__":

    # Read mnist data
    number_of_classes = 10
    number_of_training_samples_to_use = 1000
    number_of_test_samples_to_use = 100
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train_vectorized = ((X_train.reshape(X_train.shape[0], -1)).T)[:, 0:number_of_training_samples_to_use]
    y_train = y_train[0:number_of_training_samples_to_use]
    X_test_vectorized = ((X_test.reshape(X_test.shape[0], -1)).T)[:, 0:number_of_test_samples_to_use]
    y_test = y_test[0:number_of_test_samples_to_use]
    # number_of_images_to_view=16
    # test_x=X_train_vectorized[:,0:number_of_images_to_view].T.reshape((number_of_images_to_view,28,28))
    # display_images(test_x)
    input_dimensions = X_test_vectorized.shape[0]
    model = Hebbian(input_dimensions=input_dimensions, number_of_classes=number_of_classes,
                    transfer_function="Hard_limit", seed=8)
    print(model.calculate_percent_error(X_test_vectorized, y_test))
    model.initialize_all_weights_to_zeros()
    percent_error = []

    for k in range (10):
        model.train(X_train_vectorized, y_train, batch_size=300, num_epochs=2, alpha=0.1, gamma=0.1, learning="Delta")
        percent_error.append(model.calculate_percent_error(X_test_vectorized,y_test))
    print("******  Percent Error ******\n",percent_error)

    confusion_matrix=model.calculate_confusion_matrix(X_test_vectorized,y_test)
    print(np.array2string(confusion_matrix, separator=","))




    #
    # number_of_classes = 10
    # number_of_test_samples_to_use = 100
    # (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    # X_test_vectorized = ((X_test.reshape(X_test.shape[0], -1)).T)[:, 0:number_of_test_samples_to_use]
    # y_test = y_test[0:number_of_test_samples_to_use]
    # input_dimensions = X_test_vectorized.shape[0]
    #
    #
    #
    # model = Hebbian(input_dimensions=input_dimensions, number_of_classes=number_of_classes,
    #                 transfer_function="Hard_limit", seed=8)
    #
    # print(model.calculate_percent_error(X_test_vectorized, y_test))
    #
    # model = Hebbian(input_dimensions=input_dimensions, number_of_classes=number_of_classes,
    #                 transfer_function="Linear", seed=15)
    # print(model.calculate_percent_error(X_test_vectorized, y_test))
    #
    # model = Hebbian(input_dimensions=input_dimensions, number_of_classes=number_of_classes,
    #                 transfer_function="Sigmoid", seed=5)
    # print(model.calculate_percent_error(X_test_vectorized, y_test))
