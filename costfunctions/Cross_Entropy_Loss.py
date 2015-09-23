__author__ = 'theopavlakou'

import numpy as np
import scipy.misc as sp
rng = np.random

class Cross_Entropy_Loss():
    """
    The Cross entropy loss function to be used at the output of a Neural Network.
    """

    def loss(self, Y_hat, Y):
        """

        :param Y_hat:
        :param Y:
        :return:
        """
        if len(Y_hat.shape) == 2:
            temp = np.log(1 - Y_hat)
            temp[Y, np.arange(Y_hat.shape[1])] = np.log(Y_hat[Y, np.arange(Y_hat.shape[1])])
            return -np.mean(np.sum(temp, 0))

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
        Y_hat_copy = np.copy(Y_hat)
        temp = np.copy(Y_hat)
        temp[Y, np.arange(Y_hat.shape[1])] -= 1.0
        Y_hat_copy *= (1-Y_hat_copy)
        temp = temp/Y_hat_copy
        return np.mean(temp, 1)

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
        Y_hat_copy = np.copy(Y_hat)
        temp = np.copy(Y_hat)
        temp[Y, np.arange(Y_hat.shape[1])] -= 1.0
        Y_hat_copy *= (1-Y_hat_copy)
        return temp/Y_hat_copy
