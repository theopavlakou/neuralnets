__author__ = 'theopavlakou'

from Layer import Layer
import numpy as np
rng = np.random
import copy

class Neural_Net(object):
    """
    This wraps together a bunch of Layers (Layer.py) together and allows for the
    functionality of the Neural Network. It implements methods such as feeding
    forward through the network, backpropagation and getting derivatives.

    Layers are added after initialising the Neural Network and as long as
    the layers are compatible (i.e. the number of inputs for one layer is equal to
    the number of outputs of the previous layer), then as many layers as are
    desired can be stacked together.

    The only thing the Neural Network needs to know upon initialisation is the cost
    function that is to be used.
    """

    def __init__(self, cost_function, l=0.0):
        """
        Initialises the Neural_Net with the cost function that is desired.

        :param cost_function:  the cost function that is to be used to determine
                               how well the network works.
        :param l:   constant for l2 regularisation.
        """

        self.cost_function = cost_function
        self.layers = []
        self.l = l

    def num_layers(self):
        """
        :return: the number of layers in the network.
        """
        return len(self.layers)

    def add_layer(self, num_in, num_out, activation_function):
        """
        Adds a new layer to the network to come after the last layer already
        in the Neural_Net.

        :param num_in: the number of inputs for the layer.
        :param num_out: the number of outputs for the layer.
        :param activation_function: the activation function the layer should
                                    use.
        :return:    True, if successful.
                    False, if not (if layer does not align with previous).
        """
        if len(self.layers) == 0:
            self.layers.append(Layer(num_in, num_out, activation_function))
            return True
        else:
            # Need to check that the next layer will be compatible with the previous
            if self.layers[-1].num_out == num_in:
                self.layers.append(Layer(num_in, num_out, activation_function))
                return True
            else:
                print("New layer not compatible with previous layer =>")
                print("  Previous has {0} output units".format(self.layers[-1].num_out))
                print("  New has {0} input units".format(num_in))
                return False

    def feed_forward(self, x):
        """
        Feed forward through the layers and evaluate the Neural_Net at x.

        :param x:   the input to the network. Must be the same size as the input to
                    the first layer.
        :return:    the output, if the input is the right size, else, nothing.
        """
        # TODO should return something, if the input dimension is wrong.
        if len(x.shape) == 1:
            x = x.reshape((len(x), 1))

        if self.layers[0].num_in == x.shape[0]:
            dz = self.layers[0].feed_forward(x)
            for i in xrange(1, len(self.layers)):
                self.layers[i].dz = dz
                dz = self.layers[i].feed_forward(self.layers[i-1].a)
            return self.layers[-1].a
        else:
            print("    Feed_Forward: Incompatible input")

    def feed_backward(self, y_hat, y):
        """
        Go backward through the network setting up the deltas i.e. perform backpropagation.

        :param y_hat: the output of the network for the current input.
        :param y: the target for the current input.
        :return: delta for the first layer.
        """
        delta_from_forward = self.cost_function.derivative_per_data_point(y_hat, y)*self.layers[-1].activation_function.derivative(self.layers[-1].z)

        for i in xrange(len(self.layers)):
            index = self.num_layers() - 1 - i
            self.layers[index].set_delta(delta_from_forward)
            delta_from_forward = self.layers[index].feed_backward()
        # TODO do I actually need this?
        return delta_from_forward

    def feed_forward_and_backward(self, x, y):
        """
        Goes once forward and backward through the network.

        :param x: input.
        :param y: target for x.
        :return: the output of the network at x.
        """
        y_hat = self.feed_forward(x)
        self.feed_backward(y_hat, y)
        return y_hat

    def train_sgd(self, X, Y, n_iter=10000, step_size=0.005, mini_batch_size=1, test_data=None):
        """
        Train the network using SGD. This involves doing back propagation
        and calculating gradients for W and b.

        :param X:   the set of inputs. This is a (d, N) matrix, where d is
                    the dimension of the data and N is the number of data
                    points.
        :param Y:   the set of output targets. This is a (m, N) matrix,
                    where m is the output dimension and N is the number of
                    data points.
        :param mini_batch_size: the size of the minibatch to be used.
        :param n_iter:  the number of iterations of SGD.
        :param step_size:   the step size for SGD.
        :param test_data:   if test_data exists, it calculates the accuracy
                            on it. This is a list with the first term
                            being X_test and the second being Y_test
                            in exactly the same format as X and Y, resp.

        :return costs:  The costs on the training data.
        :return accuracies: The accuracies on the test data, if provided.
        """
        costs = []
        accuracies = []
        N = X.shape[1]
        self.cost_and_accuracy(X, Y, costs, accuracies, test_data)

        counter = 0
        if mini_batch_size == 0: mini_batch_size = 1
        for n in xrange(int(np.floor(n_iter/mini_batch_size))):
            # TODO Change back
            # if n > 0.95*n_iter:
            #     step_size = step_size*0.95
            i = rng.randint(N, size=mini_batch_size)

            x = X[:, i]
            if len(Y.shape) == 2:
                y = Y[:, i]
            else:
                y = Y[i]
            y_hat = self.feed_forward(x)
            self.feed_backward(y_hat, y)

            for j in xrange(self.num_layers()):
                index = self.num_layers() - 1 - j

                update_W, delta = self.get_derivative_W_and_delta_gd(index, y_hat, y, x=x, N=mini_batch_size)

                self.layers[index].W = self.layers[index].W - step_size*update_W
                self.layers[index].b = self.layers[index].b - step_size*delta
            counter += mini_batch_size

            if counter >= N:
                counter -= N
                self.cost_and_accuracy(X, Y, costs, accuracies, test_data)

        if counter > 0:
            self.cost_and_accuracy(X, Y, costs, accuracies, test_data)

        print("    First cost {0} and last cost {1}".format(costs[0], costs[-1]))
        if test_data:
            print("    First accuracy {0} and last accuracy {1}".format(accuracies[0], accuracies[-1]))
        return costs, accuracies

    def train_gd(self, X, Y, n_iter = 10000, step_size = 0.015, test_data=None):
        """
        Train the network using Gradient Descent. This involves doing back propagation
        and calculating gradients for W and b.

        :param X:   the set of inputs. This is a (d, N) matrix, where d is
                    the dimension of the data and N is the number of data
                    points.
        :param Y:   the set of output targets. This is a (m, N) matrix,
                    where m is the output dimension and N is the number of
                    data points.
        :param n_iter:  the number of stochastic gradient evaluations
                        i.e. in gradient descent this n_iter/N is the
                        number of iterations of the algorithm.
        :param test_data:   if test_data exists, it calculates the accuracy
                            on it. This is a list with the first term
                            being X_test and the second being Y_test
                            in exactly the same format as X and Y, resp.
        :param step_size:   the step size for GD.

        :return costs:  The costs on the training data.
        :return accuracies: The accuracies on the test data, if provided.
        """
        costs = []
        accuracies = []
        N = X.shape[1]
        n_epochs = int(n_iter/N)
        for n in xrange(n_epochs):
            self.cost_and_accuracy(X, Y, costs, accuracies, test_data)

            Y_hat = self.feed_forward(X)
            self.feed_backward(Y_hat, Y)

            for j in xrange(self.num_layers()):
                index = self.num_layers() - 1 - j
                update_W, delta = self.get_derivative_W_and_delta_gd(index, Y_hat, Y, x=X, N=N)
                self.layers[index].W = self.layers[index].W - step_size*update_W
                self.layers[index].b = self.layers[index].b - step_size*delta

        self.cost_and_accuracy(X, Y, costs, accuracies, test_data)
        print("    First cost {0} and last cost {1}".format(costs[0], costs[-1]))
        if test_data:
            print("    First accuracy {0} and last accuracy {1}".format(accuracies[0], accuracies[-1]))
        return costs, accuracies

    def train_svrg(self, X, Y, n_iter = 10000, step_size = 0.00001, inner_it_multiplier=5, test_data=None):
        """
        Train the network using SVRG. This involves doing back propagation
        and calculating gradients for W and b.

        :param X:   the set of inputs. This is a (d, N) matrix, where d is
                    the dimension of the data and N is the number of data
                    points.
        :param Y:   the set of output targets. This is a (m, N) matrix,
                    where m is the output dimension and N is the number of
                    data points.
        :param n_iter:  the number of iterations of SVRG.
        :param inner_it_multiplier: The number of times to do the inner
                                    inner iteration loop, in terms of the
                                    number of epochs.
        :param step_size:   the step size for SVRG.
        :param test_data:   if test_data exists, it calculates the accuracy
                            on it. This is a list with the first term
                            being X_test and the second being Y_test
                            in exactly the same format as X and Y, resp.
        :param step_size:   the step size for GD.

        :return costs:  The costs on the training data.
        :return accuracies: The accuracies on the test data, if provided.
        """
        costs = []
        accuracies = []
        N = X.shape[1]
        # As is used in the paper
        m = inner_it_multiplier*N
        done = False

        self.cost_and_accuracy(X, Y, costs, accuracies, test_data)

        counter = 0

        while not done:
            # At each outer iteration we will save a copy of the neural network with
            # its old parameters, which can be used to get the stochastic grads at
            # the old parameters. Written as w_tilde in the paper.
            old_self = copy.deepcopy(self)
            Y_hat = self.feed_forward(X)
            self.feed_backward(Y_hat, Y)
            mean_total_grad = []

            # Saves the mean gradient (mu_tilde in the paper)
            for l in xrange(self.num_layers()):
                index = self.num_layers() - 1 - l
                update_W, delta = self.get_derivative_W_and_delta_gd(index, Y_hat, Y, x=X, N=N)
                mean_total_grad.insert(0, (update_W, delta))
            n_iter -= N
            if n_iter <= 0:
                break

            # Have gone through the data once by this point, without doing any iterations.
            self.cost_and_accuracy(X, Y, costs, accuracies, test_data)

            # Option II in the paper (which is also a special case of S2GD)
            inner_it = rng.randint(m)
            for j in xrange(inner_it):

                i = rng.randint(N, size=1)
                x = X[:, i]
                if len(Y.shape) == 2:
                    y = Y[:, i]
                else:
                    y = Y[i]
                # The reason sgd and svrg takes so long is because we call feed_forward
                # so many more times.
                # TODO Replace by feed_forward_and_backward()
                y_hat_tilde = old_self.feed_forward(x)
                old_self.feed_backward(y_hat_tilde, y)

                y_hat = self.feed_forward(x)
                self.feed_backward(y_hat, y)

                for l in xrange(self.num_layers()):
                    index = self.num_layers() - 1 - l
                    update_W, delta = self.get_derivative_W_and_delta_gd(index, y_hat, y, x=x, N=1)
                    update_W_tilde, delta_tilde = old_self.get_derivative_W_and_delta_gd(index, y_hat_tilde, y, x=x, N=1)

                    self.layers[index].W = self.layers[index].W - step_size*(update_W + mean_total_grad[index][0] - update_W_tilde)
                    self.layers[index].b = self.layers[index].b - step_size*(delta + mean_total_grad[index][1] - delta_tilde)

                # One for each stochastic gradient evaluation.
                n_iter -= 2
                if n_iter <= 0:
                    done = True
                    break

                counter += 2

                if counter >= N:
                    self.cost_and_accuracy(X, Y, costs, accuracies, test_data)
                    counter = 0
        if counter > 0:
            self.cost_and_accuracy(X, Y, costs, accuracies, test_data)
        print("    First cost {0} and last cost {1}".format(costs[0], costs[-1]))
        if test_data:
            print("    First accuracy {0} and last accuracy {1}".format(accuracies[0], accuracies[-1]))
        return costs, accuracies

    def train_saga(self, X, Y, n_iter = 100, step_size = 0.00001, test_data=None):
        """
        Train the network using SVRG. This involves doing back propagation
        and calculating gradients for W and b.

        :param X:   the set of inputs. This is a (d, N) matrix, where d is
                    the dimension of the data and N is the number of data
                    points.
        :param Y:   the set of output targets. This is a (m, N) matrix,
                    where m is the output dimension and N is the number of
                    data points.
        :param n_iter:  the number of iterations of SVRG.
        :param step_size:   the step size for SVRG.
        :param test_data:   if test_data exists, it calculates the accuracy
                            on it. This is a list with the first term
                            being X_test and the second being Y_test
                            in exactly the same format as X and Y, resp.
        :param step_size:   the step size for GD.

        :return costs:  The costs on the training data.
        :return accuracies: The accuracies on the test data, if provided.
        """
        costs = []
        accuracies = []
        N = X.shape[1]
        invN = 1.0/N
        # Save all the gradients in a table.
        dW_table = []
        mean_gradients = []

        self.cost_and_accuracy(X, Y, costs, accuracies, test_data)

        for l in xrange(self.num_layers()):
            mean_gradients.append([0, 0])

        for i in xrange(N):
            x = X[:, [i]]
            if len(Y.shape) == 2:
                # Here we need to encapsulate the i in a list
                # because we are not using randint
                y = Y[:, [i]]
            else:
                y = Y[[i]]
            y_hat = self.feed_forward_and_backward(x, y)

            temp_list = []
            for l in xrange(self.num_layers()):
                index = self.num_layers() - 1 - l
                # TODO Change to gd
                update_W, delta = self.get_derivative_W_and_delta(index, y_hat, y, x)
                mean_gradients[index][0] += invN*update_W
                mean_gradients[index][1] += invN*delta
                temp_list.insert(0, (update_W, delta))

            dW_table.insert(i, temp_list)
        # We still go through the data once. So even though we haven't taken a single step
        # (i.e. the cost does not change), we still need to add it.
        self.cost_and_accuracy(X, Y, costs, accuracies, test_data)

        n_iter -= N
        counter = 0
        while n_iter > 0:
        # for n in xrange(int(n_iter)):
            i = rng.randint(X.shape[1], size=1)
            x = X[:, i]
            if len(Y.shape) == 2:
                y = Y[:, i]
            else:
                y = Y[[i]]
            y_hat = self.feed_forward_and_backward(x, y)
            old_gradients = dW_table.pop(i)

            # All the gradients for this current data point.
            temp_list = []
            # Adds new gradients to the table
            for l in xrange(self.num_layers()):
                index = self.num_layers() - 1 - l

                update_W, delta = self.get_derivative_W_and_delta_gd(index, y_hat, y, x=x, N=1)

                self.layers[index].W = self.layers[index].W - step_size*(update_W - old_gradients[index][0] + mean_gradients[index][0])
                self.layers[index].b = self.layers[index].b - step_size*(delta - old_gradients[index][1] + mean_gradients[index][1])
                mean_gradients[index][0] = mean_gradients[index][0] + invN*(update_W - old_gradients[index][0])
                mean_gradients[index][1] = mean_gradients[index][1] + invN*(delta - old_gradients[index][1])
                temp_list.insert(0, (update_W, delta))
            dW_table.insert(i, temp_list)

            n_iter -= 1
            counter += 1
            if counter >= N:
                self.cost_and_accuracy(X, Y, costs, accuracies, test_data)
                counter = 0

        print("    First cost {0} and last cost {1}".format(costs[0], costs[-1]))
        if test_data:
            print("    First accuracy {0} and last accuracy {1}".format(accuracies[0], accuracies[-1]))
        return costs, accuracies

    def train_sdca(self, X, Y, n_iter = 100, step_size = 0.00001, test_data=None):
        """
        Train the network using SDCA. This involves doing back propagation
        and calculating gradients for W and b.

        :param X:   the set of inputs. This is a (d, N) matrix, where d is
                    the dimension of the data and N is the number of data
                    points.
        :param Y:   the set of output targets. This is a (m, N) matrix,
                    where m is the output dimension and N is the number of
                    data points.
        :param n_iter:  the number of iterations of SDCA.
        :param step_size:   the step size for SDCA.
        :param test_data:   if test_data exists, it calculates the accuracy
                            on it. This is a list with the first term
                            being X_test and the second being Y_test
                            in exactly the same format as X and Y, resp.
        :param step_size:   the step size for GD.

        :return costs:  The costs on the training data.
        :return accuracies: The accuracies on the test data, if provided.
        """
        costs = []
        accuracies = []
        N = X.shape[1]

        # Initial cost and accuracy
        self.cost_and_accuracy(X, Y, costs, accuracies, test_data)
        # Initialise dual matrices/vectors
        dual_vars = []
        # Make all dual variables equal to W (or b) multiplied by lambda
        for l in xrange(self.num_layers()):
            alpha_W = self.l*np.copy(self.layers[l].W)
            alpha_b = self.l*np.copy(self.layers[l].b)
            alpha_W_list = []
            alpha_b_list = []
            for n in xrange(N):
                alpha_W_list.append(np.copy(alpha_W))
                alpha_b_list.append(np.copy(alpha_b))
            dual_vars.append((alpha_W_list, alpha_b_list))

        counter = 0
        for n in xrange(n_iter):
            i = rng.randint(N, size=1)
            # ii is the same as i but is a scalar
            ii = i[0]
            x = X[:, i]
            if len(Y.shape) == 2:
                y = Y[:, i]
            else:
                y = Y[i]
            y_hat = self.feed_forward_and_backward(x, y)

            for l in xrange(self.num_layers()):
                index = self.num_layers() - 1 - l
                dW, delta = self.get_derivative_W_and_delta_gd_no_regularisation(index, y_hat, y, x=x, N=1)

                update_W = dW + dual_vars[index][0][ii]
                update_b = delta + dual_vars[index][1][ii]

                dual_vars[index][0][ii] -= step_size*self.l*N*update_W
                dual_vars[index][1][ii] -= step_size*self.l*N*update_b

                self.layers[index].W -= step_size*update_W
                self.layers[index].b -= step_size*update_b
            counter += 1

            if counter >= N:
                self.cost_and_accuracy(X, Y, costs, accuracies, test_data)
                counter = 0
        if counter > 0:
            # If we finish and we have updated the parameters since the last cost/accuracy
            # was taken, the we add the final cost/accuracy.
            self.cost_and_accuracy(X, Y, costs, accuracies, test_data)

        print("    First cost {0} and last cost {1}".format(costs[0], costs[-1]))
        if test_data:
            print("    First accuracy {0} and last accuracy {1}".format(accuracies[0], accuracies[-1]))
        return costs, accuracies

    def get_derivative_W_and_delta(self, index, y_hat, y, x):
        """
        Get the gradient of W and of b (which is equal to delta) for the layer
        indexed by index.

        :param index: the layer number.
        :param y_hat: the output of the Neural_Net for the current input.
        :param y: the target for the current input.
        :param x: the input. This is only required if index == 0.
        :return: the update for W.
        :return: the update for b.
        """
        if (index + 1) == self.num_layers():
            delta = self.cost_function.derivative_per_data_point(y_hat, y)*self.layers[-1].activation_function.derivative(self.layers[-1].z)
        else:
            delta = self.layers[index+1].delta

        if index > 0:
            update_W = np.outer(delta, self.layers[index-1].a) + self.l * self.layers[index].W
        else:
            update_W = np.outer(delta, x) + self.l * self.layers[index].W

        # TODO Should change back to regularising bias
        # delta += self.l *self.layers[index].b

        return update_W, delta

    def get_derivative_W_and_delta_gd(self, index, y_hat, y, x, N):
        """
        Get the gradient of W and of b (which is equal to delta) for the layer
        indexed by index.

        :param index: the layer number.
        :param y_hat: the output of the Neural_Net for the current input.
        :param y: the target for the current input.
        :param x: the input. This is only required if index == 0.
        :param N: the number of data points.
        :return: the update for W.
        :return: the update for b.
        """
        if (index + 1) == self.num_layers():
            delta = self.cost_function.derivative_per_data_point(y_hat, y)*self.layers[-1].activation_function.derivative(self.layers[-1].z)
        else:
            delta = self.layers[index+1].delta

        if index > 0:
            update_W = 1.0/N*np.dot(delta, self.layers[index-1].a.T) + self.l *self.layers[index].W
        else:
            update_W = 1.0/N*np.dot(delta, x.T) + self.l *self.layers[index].W
        # TODO Should change back to regularising bias
        delta = np.reshape(np.mean(delta, 1), (delta.shape[0], 1)) #+ self.l *self.layers[index].b

        return update_W, delta

    def get_derivative_W_and_delta_gd_no_regularisation(self, index, y_hat, y, x, N):
        """
        Get the gradient of W and of b (which is equal to delta) for the layer
        indexed by index but ignoring the regularisation term. This is useful
        for SDCA at this stage.

        :param index: the layer number.
        :param y_hat: the output of the Neural_Net for the current input.
        :param y: the target for the current input.
        :param x: the input. This is only required if index == 0.
        :param N: the number of data points.
        :return: the update for W.
        :return: the update for b.
        """
        if (index + 1) == self.num_layers():
            delta = self.cost_function.derivative_per_data_point(y_hat, y)*self.layers[-1].activation_function.derivative(self.layers[-1].z)
        else:
            delta = self.layers[index+1].delta

        if index > 0:
            update_W = 1.0/N*np.dot(delta, self.layers[index-1].a.T)
        else:
            update_W = 1.0/N*np.dot(delta, x.T)
        delta = np.reshape(np.mean(delta, 1), (delta.shape[0], 1))

        return update_W, delta

    def loss(self, X, Y):
        """
        The current loss of the Neural_Net when the weights are set as they are.
        It uses l2 regularisation.

        :param X:   the set of inputs. This is a (d, N) matrix, where d is
                    the dimension of the data and N is the number of data
                    points.
        :param Y:   the set of output targets. This is a (m, N) matrix,
                    where m is the output dimension and N is the number of
                    data points.
        :return:    the loss.
        """
        l2_loss = 0.0
        if self.l > 0:
            for l in xrange(self.num_layers()):
                l2_loss += np.sum(self.layers[l].W**2)
                # TODO Change
                # l2_loss += np.sum(self.layers[l].b**2)
            l2_loss *= self.l/2
        return self.cost_function.loss(self.feed_forward(X), Y) + l2_loss

    def loss_given_output(self, Y_hat, Y):
        """
        The current loss of the Neural_Net when the weights are set as they are.
        It uses l2 regularisation.

        :param Y_hat:   the set of outputs evaluated at the inputs. This is an
                        (m, N) matrix, where m is the output dimension and N
                        is the number of data points.
        :param Y:   the set of output targets. This is an (m, N) matrix,
                    where m is the output dimension and N is the number of
                    data points.
        :return:    the loss.
        """
        l2_loss = 0.0
        if self.l > 0:
            for l in xrange(self.num_layers()):
                l2_loss += np.sum(self.layers[l].W**2)
                #TODO Change
                # l2_loss += np.sum(self.layers[l].b**2)
            l2_loss *= self.l/2
        return self.cost_function.loss(Y_hat, Y) + l2_loss

    def classification_accuracy_given_output(self, Y_hat, Y):
        """
        Returns the classification accuracy (assuming the objective
        is to classify), given the outputs, Y_hat, and the targets, Y.
        NOTE: Only for classification tasks.

        :param Y_hat:   a (c, N) matrix where c is the number of classes
                        and N is the number of data points.
        :param Y:   a (N, ) or (1, N) vector giving the targets in
                    {0, 1, ..., (c - 1)}.
        :return:    the accuracy in [0, 1].
        """
        C = np.argmax(Y_hat, 0)
        return np.mean(C==Y)


    def cost_and_accuracy(self, X, Y, costs, accuracies, test_data, verbose=True):
        """
        Appends the cost for the input given the current parameters
        and the labels to a cost list and also the accuracy at
        the current parameters to a list for the accuracies.

        :param X:   the set of inputs. This is a (d, N) matrix, where d is
                    the dimension of the data and N is the number of data
                    points.
        :param Y:   the set of output targets. This is a (m, N) matrix,
                    where m is the output dimension and N is the number of
                    data points.
        :param costs: a list to append the cost to.
        :param accuracies: a list to append the accuracy to.
        """
        Y_hat = self.feed_forward(X)
        costs.append(self.loss_given_output(Y_hat, Y))
        if verbose:
            print("        Cost is {0}".format(format(costs[-1], '.3g')))

        if test_data:
            Y_hat_test = self.feed_forward(test_data[0])
            accuracies.append(self.classification_accuracy_given_output(Y_hat_test, test_data[1]))
            if verbose:
                print("        Accuracy is {0}".format(accuracies[-1]))
