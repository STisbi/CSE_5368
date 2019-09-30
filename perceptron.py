# Tisbi, Seth-Amittai
# 1000-846-338
# 2019-09-22
# Assignment-01-02

import numpy as np
import itertools

class Perceptron(object):

    """
    Initialize Perceptron model
    :param input_dimensions: The number of features of the input data, for example (height, weight) would be two features.
    :param number_of_classes: The number of classes.
    :param seed: Random number generator seed.
    """
    def __init__(self, input_dimensions=2,number_of_classes=4,seed=None):
        if seed != None:
            np.random.seed(seed)

        self.input_dimensions  = input_dimensions
        self.number_of_classes = number_of_classes

        self._initialize_weights()


    """
    Initialize the weights, initalize using random numbers.
    Note that number of neurons in the model is equal to the number of classes
    """
    def _initialize_weights(self):
        # Create a Sx(R+1) list of random numbers from the normal distribution. The plus one in column 1 is for the bias
        weights = [[np.random.normal() for column in range(self.input_dimensions + 1)] for row in range(self.number_of_classes)]

        # Create a numpy array with this list
        self.weights = np.array(weights)

        # self.weights = np.array([[0, 1, 2],
        #                          [0, 3, 4]])

        # print("self.weights: ", self.weights.shape)


    """
    Initialize the weights, initalize using random numbers.
    """
    def initialize_all_weights_to_zeros(self):
        # Create a Sx(R+1) numpy array filled with zero's
        self.weights = np.zeros((self.number_of_classes, self.input_dimensions + 1))

        # print("Zeroed Weights: ", self.weights)

    """
    This function prints the weight matrix (Bias is included in the weight matrix).
    """
    def print_weights(self):
        for row in self.weights:
            print(row)


    """
    Given a batch of data, and the necessary hyperparameters,
    this function adjusts the self.weights using Perceptron learning rule.
    Training should be repeted num_epochs time.
    :param X: Array of input [input_dimensions,n_samples]
    :param y: Array of desired (target) outputs [number_of_classes ,n_samples]
    :param num_epochs: Number of times training should be repeated over all input data
    :param alpha: Learning rate
    :return: None
    """
    def train(self, X, Y, num_epochs=10, alpha=0.001):
        for epoch in range(num_epochs):
            # print("***************************************", epoch, "***************************************")
            # print("weights:\n", self.weights, self.weights.shape, "\nEnd Weights")

            # Cast to a numpy array
            results = np.array(self.predict(X))

            # print("Actual Results:\n", actualResults, "\nEnd Actual Results")

            # Calculate the new weights using the perceptron learning rule
            for column in range(Y.shape[1]):
                for row in range(Y.shape[0]):
                    error = Y[row][column] - results[row][column]
                    ep = np.dot(error, X[:,[column]])
                    ep_t = np.transpose(ep)
                    self.weights[row:row + 1: , 1:] = np.add(self.weights[row:row + 1: , 1:], alpha * ep_t)

            print("New Weights:\n", self.weights, "\nEnd New Weights")


    """
    Make a prediction on an array of inputs
    :param X: Array of input [input_dimensions,n_samples]. Note that the input X does not include a row of ones
    as the first row.
    :return: Array of model outputs [number_of_classes ,n_samples]
    """
    def predict(self, X):
        print("Predict Start")

        # Create a 1xR list filled with 1's
        temp = [1 for row in range(X.shape[1])]

        # Add this list (made into a numpy array) as the first row of the input matrix
        newX = np.vstack((np.array(temp), X))

        # print("newX:\n", newX, newX.shape, "\nEnd newX")

        # print("self.weights:\n", self.weights, self.weights.shape, "\nEnd self.weights")

        # Multiply the weight matrix, W, by the input matrix X
        results = np.dot(self.weights, newX)

        # results = np.dot(self.weights, X)

        # print("Results:\n", results, results.shape, "\nEnd Results")

        # Create a 2d list with the dimensions of the result from the above matrix result
        hardLim = [[0 for column in range(results.shape[1])] for row in range(results.shape[0])]

        # Calculate the hard limit
        for rowIndex, row in enumerate(results):
            for columnIndex, target in enumerate(row):
                if target <= 0:
                    hardLim[rowIndex][columnIndex] = 0
                else:
                    hardLim[rowIndex][columnIndex] = 1

        # Cast to a numpy array
        actualResults = np.array(hardLim)

        print("actualResults:\n", actualResults, "\nEnd actualResults")

        print("Predict End")

        return actualResults


    """
    Given a batch of data this function calculates percent error.
    For each input sample, if the output is not the same as the desired output, Y,
    then it is considered one error. Percent error is number_of_errors / number_of_samples.
    :param X: Array of input [input_dimensions,n_samples]
    :param y: Array of desired (target) outputs [number_of_classes ,n_samples]
    :return percent_error
    """
    def calculate_percent_error(self, X, Y):
        print("Error Start")

        results = self.predict(X)

        numErrors = 0

        for rowIndex, row in enumerate(Y):
            for columnIndex, target in enumerate(row):
                if target != results[rowIndex][columnIndex]:
                    numErrors += 1

        print("Number of Errors: %d\nNumber of samples: %d"%(numErrors, Y.shape[0] * Y.shape[1]))

        percentError = numErrors / (Y.shape[0] * Y.shape[1])

        print("Percent Error: ", percentError)

        print("Error End")

        return percentError


"""
This main program is a sample of how to run your program.
You may modify this main program as you desire.
"""
def main():
    input_dimensions = 2
    number_of_classes = 2

    model = Perceptron(input_dimensions=input_dimensions, number_of_classes=number_of_classes, seed=1)

    # model.initialize_all_weights_to_zeros()

    X_train = np.array([[-1.43815556, 0.10089809, -1.25432937, 1.48410426],
                        [-1.81784194, 0.42935033, -1.2806198, 0.06527391]])

    # X_train = np.array([[0, 0, 0, 0],
    #                     [5, 7, 1, 2],
    #                     [6, 8, 3, 4]])

    Y_train = np.array([[1, 0, 0, 1], [0, 1, 1, 0]])

    # Y_train = np.array([[1, 1, 0, 0],
    #                     [1, 1, 0, 1]])

    model.train(X_train, Y_train, num_epochs=1, alpha=0.0001)

    print("****** Model weights ******\n",model.weights)
    print("****** Input samples ******\n",X_train)
    print("****** Desired Output ******\n",Y_train)
    percent_error=[]
    for k in range (20):
        model.train(X_train, Y_train, num_epochs=1, alpha=0.0001)
        percent_error.append(model.calculate_percent_error(X_train,Y_train))
    print("******  Percent Error ******\n",percent_error)
    print("****** Model weights ******\n",model.weights)


if __name__ == "__main__":
    main()