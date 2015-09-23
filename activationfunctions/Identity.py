__author__ = 'theopavlakou'

import numpy as np

class Identity(object):
    """
    The identity activation function. Just outputs the input.
    """
    def __init__(self):
        pass

    def eval(self, X):
        """
        Outputs the input.

        :param X:   the input which is a (p, 1) vector.
        :return:    the input.
        """
        return X

    def derivative(self, X):
        """
        The derivative which is just 1 for each dimension.

        :param X:   the input which is a (p, 1) vector.
        :return:    a vector of ones with the same dimension
                    as the input.
        """
        return np.ones(X.shape)