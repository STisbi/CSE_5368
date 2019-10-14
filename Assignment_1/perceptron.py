# Tisbi, Seth-Amittai
# 1000-846-338
# 2019-09-22
# Assignment-01-02

import numpy as np

class Perceptron(object):
    weights = None
    def __init__(self, input_dimensions=2, number_of_classes=4, seed=None):
        if seed != None:
            np.random.seed(seed)

        self.input_dimensions = input_dimensions
        self.number_of_classes = number_of_classes

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

    def print_weights(self):
        print(self.weights)

    def train(self, X, Y, num_epochs=10, alpha=0.001):
        for epoch in range(num_epochs):
            for sample in range(X.shape[1]):
                # Get the nth column from both the input and expected output
                sampleX = np.transpose(np.array([X[:,sample]]))
                sampleY = np.transpose(np.array([Y[:,sample]]))

                # Get the prediction
                results = self.predict(sampleX)

                # Calculate e
                e = np.subtract(sampleY, results)

                # Calculate e dot p, where p is the input matrix
                ep = np.dot(e, np.transpose(sampleX))

                # Multiply this new matrix by the scalar alpha
                rate = np.multiply(alpha, ep)

                # Calculate the new weights without the bias
                newWeights = np.add(self.weights[:, 1:], rate)

                # The first column of the weight matrix is the bias
                bias_rate = np.multiply(alpha, e)

                # Add the old bias to this new scaled version of e
                bias = np.add(np.transpose(np.array([self.weights[:, 0]])), bias_rate)

                # Add the bias back into the weights
                self.weights = np.append(bias, newWeights, axis=1)


    def predict(self, X):
        # Add this list (made into a numpy array) as the first row of the input matrix
        newX = np.vstack((np.array([1 for column in range(X.shape[1])]), X))

        # Multiply the weight matrix, W, by the input matrix X
        results = np.dot(self.weights, newX)

        hardLimResults = np.where(results < 0, 0, 1)

        return hardLimResults

    def calculate_percent_error(self, X, Y):
        numErrors = 0

        for sample in range(X.shape[1]):
            # Get the nth column from both the input and expected output
            sampleX = np.transpose(np.array([X[:, sample]]))
            sampleY = np.transpose(np.array([Y[:, sample]]))

            results = self.predict(sampleX)

            if not np.array_equal(results, sampleY):
                    numErrors += 1

        print("Number of Errors: %d\nNumber of samples: %d"%(numErrors, Y.shape[1]))

        percentError = numErrors / Y.shape[1]

        print("Percent Error: ", percentError)

        return percentError




"""
This main program is a sample of how to run your program.
You may modify this main program as you desire.
"""
def main():
    input_dimensions = 2
    number_of_classes = 2

    perceptron = Perceptron(input_dimensions, number_of_classes, 1)

    X_train = np.array([[-1.43815556, 0.10089809, -1.25432937, 1.48410426],
                        [-1.81784194, 0.42935033, -1.2806198, 0.06527391]])

    Y_train = np.array([[1, 0, 0, 1],
                        [0, 1, 1, 0]])

    # Y_train = np.array([[1, 1, 1, 1],
    #                     [1, 0, 1, 1]])

    perceptron.initialize_all_weights_to_zeros()
    # print(perceptron.predict(X_train))
    error = []
    for k in range(20):
        perceptron.train(X_train, Y_train, num_epochs=1, alpha=0.0001)
        error.append(perceptron.calculate_percent_error(X_train, Y_train))
    print(error)


if __name__ == "__main__":
    main()