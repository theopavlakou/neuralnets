__author__ = 'theopavlakou'

import numpy as np
import scipy.misc as sp
rng = np.random

class Logistic_Loss():
    """
    The logistic loss function to be used at the output of a Neural Network.
    """

    def softmax(self, Y):
        """
        Returns the softmax of a matrix column-wise.
        :param Y:   A (m, N) matrix, where m is the dimension of the output
                    (also the number of classes) and N is the number of
                    data points.
        :return:    An (m, N) matrix with the softmax function computed on
                    each column, such that each column is normalised.
        """
        # TODO will probably experience overflows
        return np.exp(Y - sp.logsumexp(Y, 0))
        # Y = np.exp(Y)
        # Z = np.sum(Y, 0)
        # return Y/Z

    def loss(self, Y_hat, Y):
        """
        Returns the loss for the output of a Neural Network when given the targets.

        :param Y_hat:   the output of the Neural Network. An (m, N) matrix, where m
                        is the dimension of the output and N is the number of data
                        points.
        :param Y:   the targets.  An (N, 1) vector, which gives the output of the
                    target for each input. This is an integer in {0, 1, ..., m-1}.
                    N is the number of data points.
        :return: the loss which is a scalar.
        """
        if len(Y_hat.shape) == 2:
            temp = Y_hat - sp.logsumexp(Y_hat, 0)
            return -np.mean(temp[Y, np.arange(Y_hat.shape[1])])

    def derivative(self, Y_hat, Y):
        """
        The derivative for each data point.

        :param Y_hat:   the output of the Neural Network. An (m, N) matrix, where m
                        is the dimension of the output and N is the number of data
                        points.
        :param Y:   the targets.  An (N, 1) vector, which gives the output of the
                    target for each input. This is an integer in {0, 1, ..., m-1}.
                    N is the number of data points.
        :return:    the derivative which is a (m, N) matrix, where the ith column is the
                    gradient for the loss of the ith data point.
        """
        Y_hat = self.softmax(Y_hat)
        Y_hat[Y, np.arange(Y_hat.shape[1])] -= 1.0
        return np.mean(Y_hat, 1)

    def derivative_per_data_point(self, Y_hat, Y):
        """
        The derivative for each data point.

        :param Y_hat:   the output of the Neural Network. An (m, N) matrix, where m
                        is the dimension of the output and N is the number of data
                        points.
        :param Y:   the targets.  An (N, 1) vector, which gives the output of the
                    target for each input. This is an integer in {0, 1, ..., m-1}.
                    N is the number of data points.
        :return:    the derivative which is a (m, N) matrix, where the ith column is the
                    gradient for the loss of the ith data point.
        """
        Y_hat = self.softmax(Y_hat)
        Y_hat[Y, np.arange(Y_hat.shape[1])] -= 1.0
        return Y_hat