__author__ = 'theopavlakou'

import numpy as np
rng = np.random

class Squared_Loss():
    """
    The squared loss function to be used at the output of a Neural Network.
    """

    def loss(self, Y_hat, Y):
        """
        Returns the loss for the output of a Neural Network when given the targets.

        :param Y_hat:   the output of the Neural Network. An (m, N) matrix, where m
                        is the dimension of the output and N is the number of data
                        points.
        :param Y:   the targets.  An (m, N) matrix, where m
                    is the dimension of the output and N is the number of data
                    points.
        :return: the loss which is a scalar.
        """
        if len(Y_hat.shape) == 2:
            return 0.5*1.0/Y.shape[1]*np.sum((Y_hat - Y)**2)
        return 0.5*np.sum((Y_hat - Y)**2)

    def derivative(self, Y_hat, Y):
        """
        The derivative which is the mean of the derivatives for each data point.

        :param Y_hat:   the output of the Neural Network. An (m, N) matrix, where m
                        is the dimension of the output and N is the number of data
                        points.
        :param Y:   the targets.  An (m, N) matrix, where m
                    is the dimension of the output and N is the number of data
                    points.
        :return: the derivative which is a (m, 1) vector.
        """
        if len(Y_hat.shape) == 2:
            return np.column_stack(np.mean((Y_hat-Y), 1)).T
        return Y_hat - Y

    def derivative_per_data_point(self, Y_hat, Y):
        """
        The derivative for each data point.

        :param Y_hat:   the output of the Neural Network. An (m, N) matrix, where m
                        is the dimension of the output and N is the number of data
                        points.
        :param Y:   the targets.  An (m, N) matrix, where m
                    is the dimension of the output and N is the number of data
                    points.
        :return:    the derivative which is a (m, N) matrix, where the ith column is the
                    gradient for the loss of the ith data point.
        """
        return Y_hat - Y