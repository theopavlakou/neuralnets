__author__ = 'theopavlakou'

import numpy as np
rng = np.random

class Layer(object):
    """
    This is the basic module for the Neural Network. It has a number of inputs, a
    number of outputs and a weight matrix (plus bias term weights). It also needs
    to know the activation function to apply after applying the weight matrix to
    the input. The activation function, sigma, acts as such:

        a = sigma(z) = sigma(Wx + b).

    There are other terms that are useful to save when going forward and backward
    through a layer. These are:

        delta = dC/dz   which is needed by the previous layer during back
                        propagation. C is the cost function of the network.
        dz = sigma'(z)  this is needed when calculating delta and it is passed
                        forward from the previous layer.
    """

    def __init__(self, num_in, num_out, activation_function):
        """
        Initialises the layer. It needs to know the input dimension
        and the output dimension. It also needs to know the activation
        function that will be used.

        :param num_in: the dimension of the input.
        :param num_out: the dimension of the output.
        :param activation_function: the activation function to apply to z.
        """
        self.num_in = num_in
        self.num_out = num_out
        self.activation_function = activation_function
        r = np.sqrt(6.0/(num_in + num_out))
        # Should be uniform samples in [-r, r] as described in
        # http://arxiv.org/pdf/1206.5533v2.pdf.
        self.W = 2*r*(rng.random((self.num_out, self.num_in)) - 0.5)
        self.b = rng.random((num_out, 1)) - 0.5

        self.delta = rng.random(num_out)
        self.z = rng.random(num_out)
        self.a = rng.random(num_out)
        # This is equivalent to sigma'(z) in notes, where z is the output of
        # the previous layer.
        self.dz = rng.random((num_in,1))

    def feed_forward(self, x):
        """
        Feed the input, x, through the layer. It returns the gradient of the
        activation function with respect to z. The output is stored in self.a.
        This makes forward propagation easier.

        :param x:   the input. This is independent of whether this is the first
                    layer or any other layer.
        :return dz: equivalent to sigma'(z) in notes. It needs to be passed
                    forward to the next layer to ge the next delta.
        """
        self.z = np.dot(self.W, x) + self.b
        self.a = self.activation_function.eval(self.z)
        dz = self.activation_function.derivative(self.z)
        return dz

    def set_delta(self, delta_from_forward):
        """
        Sets delta for this layer given the delta for the following layer.
        delta = dC/dz.

        :param delta_from_forward: the delta from the layer in front.
        """
        temp = np.dot(self.W.T, delta_from_forward)
        self.delta = temp*self.dz

    def feed_backward(self):
        """
        :return delta: dC/dz.
        """
        return self.delta
