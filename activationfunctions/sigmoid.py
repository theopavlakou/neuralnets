__author__ = 'theopavlakou'

import numpy as np

class Sigmoid(object):
    """
    The sigmoid activation function commonly used in Neural Networks
    and used for Logistic Regression.
    """
    def __init__(self):
        pass

    def eval(self, X):
        """
        Returns the sigmoid function evaluated on every element of
        the input.

        :param X:   the input which is a (p, 1) vector.
        :return:    a (p, 1) vector with the sigmoid applied to each
                    element of X.
        """
        # TODO Guard against overflow
        return np.reshape(1.0/(1.0+np.exp(-X)), X.shape)

    def derivative(self, X):
        """
        Get the derivative of the sigmoid for each element of the input.
        Note that this is sigma'(z) = sigma(z)*sigma(-z).

        :param X:   the input which is a (p, 1) vector.
        :return:    a (p, 1) vector with the derivative of the
                    sigmoid applied to each element.
        """
        return np.reshape(self.eval(X)*self.eval(-X), X.shape)