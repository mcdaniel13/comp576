__author__ = 'tan_nguyen'

import numpy as np
from sklearn import datasets, linear_model
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def generate_data():
    '''
    generate data
    :return: X: input data, y: given labels
    '''
    np.random.seed(0)
    X, y = datasets.make_moons(200, noise=0.20)
    return X, y

def generate_other_data():
    '''
    generate other data
    :return: X: input data, y: given labels
    '''
    np.random.seed(0)
    X, y = datasets.make_circles(n_samples=200)
    return X, y

def plot_decision_boundary(pred_func, X, y):
    '''
    plot the decision boundary
    :param pred_func: function used to predict the label
    :param X: input data
    :param y: given labels
    :return:
    '''
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.show()


########################################################################################################################
########################################################################################################################
# YOUR ASSSIGMENT STARTS HERE
# FOLLOW THE INSTRUCTION BELOW TO BUILD AND TRAIN A 3-LAYER NEURAL NETWORK
########################################################################################################################
########################################################################################################################
class DeepNeuralNetwork(object):
    """
    This class builds and trains a neural network
    """

    def __init__(self, nn_dim_list, actFun_type='tanh', reg_lambda=0.01, seed=0):
        '''
        :param nn_dim_list: list of dimensions of layers
        :param actFun_type: type of activation function. 3 options: 'tanh', 'sigmoid', 'relu'
        :param reg_lambda: regularization coefficient
        :param seed: random seed
        '''
        self.nn_dim_list = nn_dim_list
        self.actFun_type = actFun_type
        self.reg_lambda = reg_lambda

        # initialize the weights and biases in the network
        self.W = []
        self.b = []

        np.random.seed(seed)
        for i in range(len(nn_dim_list) - 1):
            self.W.append(np.random.randn(self.nn_dim_list[i], self.nn_dim_list[i + 1]) / np.sqrt(self.nn_dim_list[i]))
            self.b.append(np.zeros((1, self.nn_dim_list[i + 1])))

    def actFun(self, z, type):
        '''
        actFun computes the activation functions
        :param z: net input
        :param type: Tanh, Sigmoid, or ReLU
        :return: activations
        '''

        # YOU IMPLMENT YOUR actFun HERE
        res = None
        if type == 'tanh':
            res = np.tanh(z)
        elif type == 'sigmoid':
            res = 1 / (1 + np.exp(-z))
        elif type == 'relu':
            res = np.maximum(0, z)
        return res

    def diff_actFun(self, z, type):
        '''
        diff_actFun compute the derivatives of the activation functions wrt the net input
        :param z: net input
        :param type: Tanh, Sigmoid, or ReLU
        :return: the derivatives of the activation functions wrt the net input
        '''

        # YOU IMPLEMENT YOUR diff_actFun HERE
        res = None
        if type == 'tanh':
            res = 1 - np.power(np.tanh(z), 2)
        elif type == 'sigmoid':
            sig = 1 / (1 + np.exp(-z))
            res = sig * (1 - sig)
        elif type == 'relu':
            res = 1 * (z > 0)
        return res

    def feedforward(self, X, actFun):
        '''
        feedforward builds a 3-layer neural network and computes the two probabilities,
        one for class 0 and one for class 1
        :param X: input data
        :param actFun: activation function
        :return:
        '''

        # YOU IMPLEMENT YOUR feedforward HERE
        self.z = []
        self.a = []
        dim_len = len(self.W)
        for i in range(dim_len):
            if i == 0:
                self.z.append(np.dot(X, self.W[i]) + self.b[i])
            else:
                self.z.append(np.dot(self.a[i - 1], self.W[i]) + self.b[i])

            if i != dim_len - 1:
                self.a.append(actFun(self.z[i]))

        self.probs = np.exp(self.z[len(self.z) - 1]) / np.sum(np.exp(self.z[len(self.z) - 1]), axis=1, keepdims=True)
        return None

    def calculate_loss(self, X, y):
        '''
        calculate_loss compute the loss for prediction
        :param X: input data
        :param y: given labels
        :return: the loss for prediction
        '''
        num_examples = len(X)
        self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
        # Calculating the loss

        # YOU IMPLEMENT YOUR CALCULATION OF THE LOSS HERE

        data_loss = np.sum(-np.log(self.probs[range(num_examples), y]))

        # Add regulatization term to loss (optional)
        w_sums = 0
        dim_len = len(self.W)
        for i in range(dim_len):
            w_sums += np.sum(np.square(self.W[i]))
        data_loss += self.reg_lambda / 2 * w_sums
        return (1. / num_examples) * data_loss

    def predict(self, X):
        '''
        predict infers the label of a given data point X
        :param X: input data
        :return: label inferred
        '''
        self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
        return np.argmax(self.probs, axis=1)

    def backprop(self, X, y):
        '''
        backprop run backpropagation to compute the gradients used to update the parameters in the backward step
        :param X: input data
        :param y: given labels
        :return: dL/dW1, dL/b1, dL/dW2, dL/db2
        '''

        # IMPLEMENT YOUR BACKPROP HERE
        num_example = len(X)
        del3 = self.probs
        del3[range(num_example), y] -= 1

        dW = []
        db = []
        z_len = len(self.z)
        for i in range(z_len):
            j = z_len - 1 - i
            if j == 0:
                dW.insert(0, np.dot(X.T, del3))
                db.insert(0, np.sum(del3, axis=0, keepdims=False))
            else:
                dW.insert(0, np.dot(self.a[j - 1].T, del3))
                db.insert(0, np.sum(del3, axis=0, keepdims=True))
                diff = self.diff_actFun(self.z[j - 1], type=self.actFun_type)
                del3 = diff * np.dot(del3, self.W[j].T)

        return dW, db

    def fit_model(self, X, y, epsilon=0.01, num_passes=20000, print_loss=True):
        '''
        fit_model uses backpropagation to train the network
        :param X: input data
        :param y: given labels
        :param num_passes: the number of times that the algorithm runs through the whole dataset
        :param print_loss: print the loss or not
        :return:
        '''
        # Gradient descent.
        for i in range(0, num_passes):
            # Forward propagation
            self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
            # Backpropagation
            dW, db = self.backprop(X, y)

            # Add derivatives of regularization terms (b1 and b2 don't have regularization terms)
            for i in range(len(dW)):
                dW[i] += self.reg_lambda * self.W[i]

            # Gradient descent parameter update
            dim_len = len(self.W)
            for i in range(dim_len):
                self.W[i] += -epsilon * dW[i]
                self.b[i] += -epsilon * db[i]

            # Optionally print the loss.
            # This is expensive because it uses the whole dataset, so we don't want to do it too often.
            if print_loss and i % 1000 == 0:
                print("Loss after iteration %i: %f" % (i, self.calculate_loss(X, y)))

    def visualize_decision_boundary(self, X, y):
        '''
        visualize_decision_boundary plot the decision boundary created by the trained network
        :param X: input data
        :param y: given labels
        :return:
        '''
        plot_decision_boundary(lambda x: self.predict(x), X, y)


def main():
    # generate and visualize Make-Moons dataset
    X, y = generate_other_data()
    # plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
    # plt.show()

    model = DeepNeuralNetwork(nn_dim_list=[2, 4, 4, 2], actFun_type='sigmoid')
    model.fit_model(X, y)
    model.visualize_decision_boundary(X, y)


if __name__ == "__main__":
    main()
