__author__ = 'theopavlakou'

import unittest
import numpy as np
import Neural_Net as nn
import costfunctions.Squared_Loss as sl
import costfunctions.Logistic_Loss as ll
import costfunctions.Cross_Entropy_Loss as cel

import activationfunctions.Identity as identity
import activationfunctions.sigmoid as sigmoid

class TestNeuralNets(unittest.TestCase):

    def setUp(self):
        cost_function = sl.Squared_Loss()
        # cost_function = cel.Cross_Entropy_Loss()
        # cost_function = ll.Logistic_Loss()
        # TODO must actually change the regularisation constant
        self.nn = nn.Neural_Net(cost_function, l=0.0001)
        self.x = np.array([1,2,3,4])

    def test_add_layer_does_not_work_for_dimensions_that_do_not_match(self):
        n_in_1 = 4
        n_out_1 = 5
        n_in_2 = 5
        n_out_2 = 4
        n_in_3 = 5
        n_out_3 = 10

        self.assertEqual(self.nn.add_layer(n_in_1, n_out_1, identity.Identity()), True)
        self.assertEqual(self.nn.add_layer(n_in_2, n_out_2, identity.Identity()), True)
        self.assertEqual(self.nn.add_layer(n_in_3, n_out_3, identity.Identity()), False)

    #######################
    ## I/O Tests
    #######################

    def test_feed_forward(self):
        n_in_1 = 4
        n_out_1 = 5
        n_in_2 = 5
        n_out_2 = 1

        self.nn.add_layer(n_in_1, n_out_1, identity.Identity())
        self.nn.add_layer(n_in_2, n_out_2, identity.Identity())
        print(self.nn.feed_forward(np.array([[1],[2],[3],[4]])))

    def test_feed_forward_two_inputs(self):
        n_in_1 = 4
        n_out_1 = 5
        n_in_2 = 5
        n_out_2 = 1

        self.nn.add_layer(n_in_1, n_out_1, identity.Identity())
        self.nn.add_layer(n_in_2, n_out_2, identity.Identity())
        print(self.nn.feed_forward(np.array([[1,2,3,4], [5,6,7,8]]).T))

    def test_feed_forward_two_inputs_sigmoid(self):
        n_in_1 = 4
        n_out_1 = 5
        n_in_2 = 5
        n_out_2 = 1

        self.nn.add_layer(n_in_1, n_out_1, sigmoid.Sigmoid())
        self.nn.add_layer(n_in_2, n_out_2, sigmoid.Sigmoid())
        print(self.nn.feed_forward(np.array([[1,2,3,4], [5,6,7,8]]).T))

    def test_feed_backward(self):
        n_in_1 = 4
        n_out_1 = 5
        n_in_2 = 5
        n_out_2 = 2

        self.nn.add_layer(n_in_1, n_out_1, identity.Identity())
        self.nn.add_layer(n_in_2, n_out_2, identity.Identity())
        y_hat = self.nn.feed_forward(self.x)

        print(self.nn.feed_backward(y_hat, self.x))

    #######################
    ## Training Tests
    #######################

    def test_train_3_layers(self, activation_function="sigmoid", algorithm="gd", iterations=100000, step_size=0.01, N=300, scale_output=True):
        print("=== Running test_train_3_layers with algorithm {0} and N={1} and activation function {2} === ".format(algorithm, N, activation_function))
        n_in_1 = 4
        n_out_1 = 5
        n_in_2 = 5
        n_out_2 = 6
        n_in_3 = 6
        n_out_3 = 4

        self.nn.cost_function = sl.Squared_Loss()

        if activation_function == "sigmoid":
            self.nn.add_layer(n_in_1, n_out_1, sigmoid.Sigmoid())
            self.nn.add_layer(n_in_2, n_out_2, sigmoid.Sigmoid())
        elif activation_function == "identity":
            self.nn.add_layer(n_in_1, n_out_1, identity.Identity())
            self.nn.add_layer(n_in_2, n_out_2, identity.Identity())
        self.nn.add_layer(n_in_3, n_out_3, identity.Identity())

        # All it does is make everything relatively small.
        X = np.random.random((4, N))
        X[:, 0] = np.array([0.8, 0.3, 0.1, 0.7])
        if scale_output == True:
            output_scaling_factor = np.array([[100, 10, 1, 0.1]]).T
        else:
            output_scaling_factor = np.array([[1, 1, 1, 1]]).T

        print("    Input is \n    {0}\n".format(X[:,0].flatten()))
        print("    Target is \n    {0}\n    ".format((X[:,0]*output_scaling_factor.flatten()).flatten()))
        print("    Evaluation before training: \n    {0}\n".format(self.nn.feed_forward(X[:, 0]).flatten()))
        print("    Using a step size of {0} and with {1} iterations and a regularisation constant of {2}.\n".format(step_size, iterations, self.nn.l))

        if algorithm == "gd":
            costs = self.nn.train_gd(X, X*output_scaling_factor, iterations, step_size=step_size)
        elif algorithm == "sgd":
            costs = self.nn.train_sgd(X, X*output_scaling_factor, iterations, step_size=step_size)
        elif algorithm == "saga":
            costs = self.nn.train_saga(X, X*output_scaling_factor, iterations, step_size=step_size)
        elif algorithm == "svrg":
            costs = self.nn.train_svrg(X, X*output_scaling_factor, iterations, step_size=step_size)
        elif algorithm == "sdca":
            costs = self.nn.train_sdca(X, X*output_scaling_factor, iterations, step_size=step_size)
        print("    Length of costs is {0}".format(len(costs)))
        # Should get something close to the input
        print("    Evaluation after training: \n    {0}\n".format(self.nn.feed_forward(X[:, 0]).flatten()))

    def test_train_3_layers_log_loss(self, algorithm="gd", iterations=100000, step_size=0.01, N=300):
        print("=== Running test_train_3_layers with algorithm {0} and N={1} === ".format(algorithm, 4*np.floor(N/4.0)))
        n_in_1 = 4
        n_out_1 = 5
        n_in_2 = 5
        n_out_2 = 6
        n_in_3 = 6
        n_out_3 = 4
        self.nn.cost_function = ll.Logistic_Loss()
        self.nn.add_layer(n_in_1, n_out_1, sigmoid.Sigmoid())
        self.nn.add_layer(n_in_2, n_out_2, sigmoid.Sigmoid())
        self.nn.add_layer(n_in_3, n_out_3, identity.Identity())

        n_per_class = int(np.floor(N/4.0))
        X_0 = np.random.random((4, n_per_class)) + 3*np.array([[1,1,1,1]]).T
        Y_0 = np.zeros((n_per_class, ))
        X_1 = np.random.random((4, n_per_class)) - 3*np.array([[1,1,1,1]]).T
        Y_1 = np.ones((n_per_class, ))
        X_2 = np.random.random((4, n_per_class)) + 3*np.array([[-1,1,-1,1]]).T
        Y_2 = 2*np.ones((n_per_class, ))
        X_3 = np.random.random((4, n_per_class)) + 3*np.array([[1,-1,1,-1]]).T
        Y_3 = 3*np.ones((n_per_class, ))

        X = np.hstack((X_0, X_1, X_2, X_3))
        Y = np.hstack((Y_0, Y_1, Y_2, Y_3)).astype(int)

        print("    Input is \n    {0}\n".format(X[: ,0].flatten()))
        print("    Target is \n    {0}\n    ".format(Y[0]))
        print("    Evaluation before training: \n    {0}\n".format(self.nn.feed_forward(X[:, 0]).flatten()))
        print("    Using a step size of {0} and with {1} iterations and a regularisation constant of {2}.\n".format(step_size, iterations, self.nn.l))

        if algorithm == "gd":
            costs = self.nn.train_gd(X, Y, iterations, step_size=step_size)
        elif algorithm == "sgd":
            costs = self.nn.train_sgd(X, Y, iterations, step_size=step_size)
        elif algorithm == "saga":
            costs = self.nn.train_saga(X, Y, iterations, step_size=step_size)
        elif algorithm == "svrg":
            costs = self.nn.train_svrg(X, Y, iterations, step_size=step_size)
        elif algorithm == "sdca":
            costs = self.nn.train_sdca(X, Y, iterations, step_size=step_size)
        print("    Length of costs is {0}".format(len(costs)))
        # Should get something close to the input
        print("    Evaluation after training: \n    {0}\n".format(self.nn.feed_forward(X[:, 0]).flatten()))

    def test_train_2_layers(self):
        n_in_1 = 4
        n_out_1 = 5
        n_in_2 = 5
        n_out_2 = 4

        self.nn.add_layer(n_in_1, n_out_1, identity.Identity())
        self.nn.add_layer(n_in_2, n_out_2, identity.Identity())
        # X = np.array([[1,2,3], [4, 5,6], [7, 8, 9], [10, 11, 12]])
        X = np.random.random((4, 10000))
        # Should get something pretty random
        print("Input is {0}".format(X[:,1].flatten()))
        print("Evaluation before training: {0}".format(self.nn.feed_forward(X[:, 1])))
        self.nn.train_sgd(X, X, 10000)
        # Should get something close to the input
        print("Evaluation after training: {0}".format(self.nn.feed_forward(X[:, 1])))
        print("Final matrix of weights is: \n {0}".format(np.dot(self.nn.layers[0].W, self.nn.layers[1].W)))
        print("Final bias terms: \n {0} and {1}".format(self.nn.layers[0].b, self.nn.layers[1].b))

    def test_train_2_layers_sgd_minibatch(self):
        n_in_1 = 4
        n_out_1 = 5
        n_in_2 = 5
        n_out_2 = 4

        self.nn.add_layer(n_in_1, n_out_1, identity.Identity())
        self.nn.add_layer(n_in_2, n_out_2, identity.Identity())
        # X = np.array([[1,2,3], [4, 5,6], [7, 8, 9], [10, 11, 12]])
        X = np.random.random((4, 100))
        # Should get something pretty random
        print("Input is {0}".format(X[:,1].flatten()))
        print("Evaluation before training: {0}".format(self.nn.feed_forward(X[:, 1]).T))
        self.nn.train_sgd(X, X, 10000, mini_batch_size=6, step_size=0.05)
        # Should get something close to the input
        print("Evaluation after training: {0}".format(self.nn.feed_forward(X[:, 1]).T))
        print("Final matrix of weights is: \n {0}".format(np.dot(self.nn.layers[0].W, self.nn.layers[1].W)))
        print("Final bias terms: \n {0} and {1}".format(self.nn.layers[0].b.T, self.nn.layers[1].b.T))

    def test_train_3_layers_sigmoid_sgd_multiple_data_points(self):
        """
            === Running test_train_3_layers with algorithm sgd and N=300 and activation function sigmoid ===
                Input is
                [ 0.8  0.3  0.1  0.7]

                Target is
                [  8.00000000e+01   3.00000000e+00   1.00000000e-01   7.00000000e-02]

                Evaluation before training:
                [ 3.66785833  3.86223013  3.81620931  3.93960815]

                Using a step size of 0.01 and with 100000 iterations and a regularisation constant of 0.0.

                Evaluation after training:
                [  7.98543636e+01   2.76904277e+00   5.53992949e-01   4.94884279e-02]
        """
        self.test_train_3_layers(algorithm="sgd")

    def test_train_3_layers_sigmoid_sgd_one_data_point(self):
        """
            === Running test_train_3_layers with algorithm sgd and N=1 and activation function sigmoid ===
                Input is
                [ 0.8  0.3  0.1  0.7]

                Target is
                [  8.00000000e+01   3.00000000e+00   1.00000000e-01   7.00000000e-02]

                Evaluation before training:
                [ 2.50107941  3.34715293  4.08044183  2.38787532]

                Using a step size of 0.01 and with 100000 iterations and a regularisation constant of 0.0.

                Evaluation after training:
                [  8.00000000e+01   3.00000000e+00   1.00000000e-01   7.00000000e-02]
        """
        self.test_train_3_layers(algorithm="sgd", N=1)

    def test_train_3_layers_sigmoid_gd_multiple_data_points(self):
        """
            === Running test_train_3_layers with algorithm gd and N=50 and activation function sigmoid ===
                Input is
                [ 0.8  0.3  0.1  0.7]

                Target is
                [ 0.8  0.3  0.1  0.7]

                Evaluation before training:
                [ 1.28171929  2.44121612  1.9635041   2.90078079]

                Using a step size of 0.015 and with 10000000.0 iterations and a regularisation constant of 0.0.

                First cost 6.27632168219 and last cost 0.000179732051851

                Evaluation after training:
                [ 0.80164898  0.29103119  0.08966104  0.70109876]
        """
        self.test_train_3_layers(activation_function="sigmoid", algorithm="gd", step_size=0.015, iterations=int(1e7), N=50, scale_output=False)

    def test_train_3_layers_sigmoid_gd_lots_of_data(self):
        """
            === Running test_train_3_layers with algorithm gd and N=100000 and activation function sigmoid ===
                Input is
                [ 0.8  0.3  0.1  0.7]

                Target is
                [ 0.8  0.3  0.1  0.7]

                Evaluation before training:
                [-0.16259908 -0.34249845  0.98536638 -0.76978109]

                Using a step size of 0.015 and with 3000000000 iterations and a regularisation constant of 0.001.

                First cost 1.67516255113 and last cost 0.0992815208822

                Length of costs is 30001

                Evaluation after training:
                [ 0.67507709  0.32248916  0.30168785  0.81601692]
        """
        self.test_train_3_layers(activation_function="sigmoid", algorithm="gd", step_size=0.015, iterations=int(3*1e9), N=int(1e5), scale_output=False)

    def test_train_3_layers_sigmoid_gd_one_data_point(self):
        """
            === Running test_train_3_layers with algorithm gd and N=1 and activation function sigmoid ===
                Input is
                [ 0.8  0.3  0.1  0.7]

                Target is
                [  8.00000000e+01   3.00000000e+00   1.00000000e-01   7.00000000e-02]

                Evaluation before training:
                [ 2.91265288  4.12040599  3.12481062  2.71073898]

                Using a step size of 0.015 and with 1000000 iterations and a regularisation constant of 0.0.

                First cost 2979.9186887 and last cost 1.61561794877e-27

                Evaluation after training:
                [  8.00000000e+01   3.00000000e+00   1.00000000e-01   7.00000000e-02]
        """
        self.test_train_3_layers(algorithm="gd", activation_function="sigmoid", N=1, step_size=0.015, iterations=int(1e6), scale_output=True)

    def test_train_3_layers_identity_gd_one_data_point(self):
        """
            === Running test_train_3_layers with algorithm gd and N=1 and activation function identity ===
                Input is
                [ 0.8  0.3  0.1  0.7]

                Target is
                [ 0.8  0.3  0.1  0.7]

                Evaluation before training:
                [ 13.72462294  18.76036302  12.02311227   8.3338857 ]

                Using a step size of 0.015 and with 100000 iterations and a regularisation constant of 0.0.

                First cost 354.133849057 and last cost 8.78224054641e-32

                Evaluation after training:
                [ 0.8  0.3  0.1  0.7]
        """
        self.test_train_3_layers(activation_function="identity", algorithm="gd", N=1, step_size=0.015, scale_output=False)

    def test_train_3_layers_identity_gd_multiple_data_points(self):
        """
            === Running test_train_3_layers with algorithm gd and N=300 and activation function identity ===
                Input is
                [ 0.8  0.3  0.1  0.7]

                Target is
                [ 0.8  0.3  0.1  0.7]

                Evaluation before training:
                [ 15.38798937  16.66666664  22.05213707  13.86617997]

                Using a step size of 0.015 and with 10000000.0 iterations and a regularisation constant of 0.0.

                Evaluation after training:
                [ 0.8  0.3  0.1  0.7]

                Final weight matrix is:
                [[  1.00000000e+00  -1.68615122e-14  -4.02455846e-15   3.51108032e-15]
                 [ -6.30051566e-15   1.00000000e+00  -1.22957200e-14  -1.01932351e-14]
                 [  2.77555756e-16  -1.17267307e-14   1.00000000e+00   5.75234305e-15]
                 [ -8.34055047e-15  -1.29583844e-14  -1.85060300e-14   1.00000000e+00]]
                Final b for layer 0 is:
                [[ 0.05375941]
                 [ 0.50613146]
                 [ 0.00752634]
                 [-0.06904292]
                 [ 0.43074728]]
                Final b for layer 1 is:
                [[-0.52245378]
                 [ 0.18776885]
                 [ 0.18596799]
                 [ 0.12451409]
                 [-0.47991051]
                 [-0.11899894]]
                Final b for layer 2 is:
                [[ 0.23091052]
                 [-0.21759451]
                 [-0.38771204]
                 [ 0.39044679]]
        """
        self.test_train_3_layers(activation_function="identity", algorithm="gd", step_size=0.015, iterations=1e7, scale_output=False)
        # Should get something similar to the identity matrix
        print("    Final weight matrix is:")
        print(np.dot(self.nn.layers[2].W, np.dot(self.nn.layers[1].W, self.nn.layers[0].W)))
        print("    Final b for layer 0 is:")
        print(self.nn.layers[0].b)
        print("    Final b for layer 1 is:")
        print(self.nn.layers[1].b)
        print("    Final b for layer 2 is:")
        print(self.nn.layers[2].b)

    def test_train_3_layers_sigmoid_gd_multiple_data_points_log_loss(self):
        """
            === Running test_train_3_layers with algorithm gd and N=1000.0 ===
                Input is
                [ 3.71088256  3.46358375  3.56304939  3.37853322]

                Target is
                0

                Evaluation before training:
                [-1.1805444  -0.59635209 -0.23682303  0.94431233]

                Using a step size of 0.015 and with 10000000 iterations and a regularisation constant of 0.0.

                First cost 1.66432726021 and last cost 0.032511804894
                Length of costs is 10001
                Evaluation after training:
                [ 3.69904609 -4.15248994 -0.31756663 -0.39075501]

                NOTE: Not soft maxed but obviously this would give the right result.
                To get the classification error, just take the max.
        """
        self.test_train_3_layers_log_loss(algorithm="gd", step_size=0.015, iterations=int(1e7), N=1000)


    def test_train_3_layers_sigmoid_svrg(self):
        """
            === Running test_train_3_layers with algorithm svrg and N=1000000 and activation function sigmoid ===
                Input is
                [ 0.8  0.3  0.1  0.7]

                Target is
                [ 0.8  0.3  0.1  0.7]

                Evaluation before training:
                [-0.31951637  1.34676381  0.74149719 -0.35117287]

                Using a step size of 0.2 and with 100000000 iterations and a regularisation constant of 0.001.

                First cost 1.26729369636 and last cost 0.040987364032

                Length of costs is 101

                Evaluation after training:
                [ 0.78026843  0.31662675  0.13679782  0.69675799]
        """
        self.test_train_3_layers(activation_function="sigmoid", algorithm="svrg", step_size=0.2, iterations=int(1e8), scale_output=False, N=int(1e6))

    def test_train_3_layers_identity_svrg(self):
        """
            === Running test_train_3_layers with algorithm svrg and N=300 and activation function identity ===
                Input is
                [ 0.8  0.3  0.1  0.7]

                Target is
                [ 0.8  0.3  0.1  0.7]

                Evaluation before training:
                [ 17.01428669  15.12529268  11.21324982  13.38360947]

                Using a step size of 0.0005 and with 1000000.0 iterations and a regularisation constant of 0.0.

                First cost 426.984084192 and last cost 1.31244485007e-18

                Evaluation after training:
                [ 0.8  0.3  0.1  0.7]
        """
        self.test_train_3_layers(activation_function="identity", algorithm="svrg", step_size=0.0005, iterations=1e6, scale_output=False)

    def test_train_3_layers_sigmoid_saga_scaled_output(self):
        """
            === Running test_train_3_layers with algorithm saga and N=300 and activation function sigmoid ===
                Input is
                [ 0.8  0.3  0.1  0.7]

                Target is
                [  8.00000000e+01   3.00000000e+00   1.00000000e-01   7.00000000e-02]

                Evaluation before training:
                [ 3.59887587  3.21030046  3.674527    3.35652201]

                Using a step size of 0.01 and with 1000000.0 iterations and a regularisation constant of 0.0.

                First cost 1468.60690952 and last cost 2.51925754328

                Evaluation after training:
                [  7.74176324e+01   3.43214079e+00   5.41576193e-01   4.81199058e-02]
        """
        self.test_train_3_layers(activation_function="sigmoid", algorithm="saga", step_size=0.01, iterations=1e6, scale_output=True)

    def test_train_3_layers_sigmoid_saga(self):
        """
            === Running test_train_3_layers with algorithm saga and N=10000 and activation function sigmoid ===
                Input is
                [ 0.8  0.3  0.1  0.7]

                Target is
                [ 0.8  0.3  0.1  0.7]

                Evaluation before training:
                [-0.72004108 -0.41500831  0.92322578 -0.35212965]

                Using a step size of 0.01 and with 100000000 iterations and a regularisation constant of 0.001.

                First cost 1.81826744516 and last cost 0.0410938010299
                Length of costs is 10001
                Evaluation after training:
                [ 0.77369945  0.31130913  0.13108072  0.69019241]
        """
        self.test_train_3_layers(activation_function="sigmoid", algorithm="saga", step_size=0.01, iterations=int(1e6), scale_output=False)

    def test_train_3_layers_sigmoid_sdca_scaled_output(self):
        """
            === Running test_train_3_layers with algorithm sdca and N=300 and activation function sigmoid ===
                Input is
                [ 0.8  0.3  0.1  0.7]

                Target is
                [  8.00000000e+01   3.00000000e+00   1.00000000e-01   7.00000000e-02]

                Evaluation before training:
                [ 4.13948073  1.99090863  3.34870786  2.06495564]

                Using a step size of 0.01 and with 1000000 iterations and a regularisation constant of 0.001.

                First cost 1633.47988792 and last cost 1.8608050718
                Evaluation after training:
                [  7.97185065e+01   2.67176771e+00   4.88432077e-01   4.46913450e-02]
        """
        self.test_train_3_layers(activation_function="sigmoid", algorithm="sdca", step_size=0.01, iterations=int(1e6), scale_output=True)

    def test_train_3_layers_sigmoid_sdca(self):
        """
            === Running test_train_3_layers with algorithm sdca and N=100000 and activation function sigmoid ===
                Input is
                [ 0.8  0.3  0.1  0.7]

                Target is
                [ 0.8  0.3  0.1  0.7]

                Evaluation before training:
                [ 3.29066592  4.50896335  4.10183365  2.86832836]

                Using a step size of 0.01 and with 1e7 iterations and a regularisation constant of 0.001.

                First cost 21.3336158109 and last cost 0.0411619731333
                Evaluation after training:
                [ 0.77801346  0.3179327   0.12476994  0.68563615]
        """
        self.test_train_3_layers(activation_function="sigmoid", algorithm="sdca", step_size=0.01, iterations=int(1e7), scale_output=False, N=int(1e5))

    def test_train_3_layers_identity_gd_multiple_data_points_all_layers_same_dimension_all_W_are_identity_matrix(self):
        # Works: first and final cost is zero.
        n_in_1 = 4
        n_out_1 = 4
        n_in_2 = 4
        n_out_2 = 4
        n_in_3 = 4
        n_out_3 = 4

        self.nn.add_layer(n_in_1, n_out_1, identity.Identity())
        self.nn.add_layer(n_in_2, n_out_2, identity.Identity())
        self.nn.add_layer(n_in_3, n_out_3, identity.Identity())
        for layer in self.nn.layers:
            layer.W = np.eye(4)
            layer.b = np.zeros((4, 1))

        X = np.random.random((4, 10))
        # Should get something pretty random
        print("Input is {0}".format(X[:,0].flatten()))
        print("Evaluation before training: {0}".format(self.nn.feed_forward(X[:, 0])))
        self.nn.train_gd(X, X, 100000, step_size=0.0001)
        # Should get something close to the input
        print("Evaluation after training: {0}".format(self.nn.feed_forward(X[:, 0])))

    #######################
    ## Derivative Tests
    #######################

    def test_derivative_from_backpropagation_identity_one_data_point(self):
        n_in_1 = 4
        n_out_1 = 5
        n_in_2 = 5
        n_out_2 = 10
        n_in_3 = 10
        n_out_3 = 6
        n_in_4 = 6
        n_out_4 = 4

        self.nn.cost_function = sl.Squared_Loss()
        self.nn.add_layer(n_in_1, n_out_1, identity.Identity())
        self.nn.add_layer(n_in_2, n_out_2, identity.Identity())
        self.nn.add_layer(n_in_3, n_out_3, identity.Identity())
        self.nn.add_layer(n_in_4, n_out_4, identity.Identity())
        X = np.random.random((n_in_1,1))
        C = self.nn.loss(X, X)
        eps = 0.000001

        y_hat = self.nn.feed_forward_and_backward(X, X)

        layer = np.random.randint(4)
        print("Layer is {0}".format(layer))

        w_index_row = np.random.randint(self.nn.layers[layer].W.shape[0])
        w_index_col = np.random.randint(self.nn.layers[layer].W.shape[1])
        b_index = np.random.randint(self.nn.layers[layer].b.shape[0])

        print("Element of W is ({0}, {1})".format(w_index_row, w_index_col))
        print("Element of b is {0}\n".format(b_index))
        # Get derivative of W
        dW, d = self.nn.get_derivative_W_and_delta(layer, y_hat, X, X)

        self.nn.layers[layer].W[w_index_row,w_index_col] = self.nn.layers[layer].W[w_index_row,w_index_col] + eps
        loss_1_W = self.nn.loss(X, X)
        self.nn.layers[layer].W[w_index_row,w_index_col] = self.nn.layers[layer].W[w_index_row,w_index_col] - 2*eps
        loss_2_W = self.nn.loss(X, X)

        fd_W = (loss_1_W - loss_2_W)/(2*eps)
        print("FD for W {0}".format(fd_W))
        print("Actual for W {0}".format(dW[w_index_row,w_index_col]))
        relative_error_W = np.abs(fd_W - dW[w_index_row, w_index_col])/(np.abs(dW[w_index_row, w_index_col]) + np.abs(fd_W))
        print("Relative error for W is {0}\n".format(relative_error_W))
        self.assertEqual( relative_error_W < 1e-5, True)
        #Need to reset this
        self.nn.layers[layer].W[w_index_row,w_index_col] = self.nn.layers[layer].W[w_index_row,w_index_col] + eps

        self.nn.layers[layer].b[b_index] = self.nn.layers[layer].b[b_index] + eps
        loss_1_b = self.nn.loss(X, X)
        self.nn.layers[layer].b[b_index] = self.nn.layers[layer].b[b_index] - 2*eps
        loss_2_b = self.nn.loss(X, X)
        self.nn.layers[layer].b[b_index] = self.nn.layers[layer].b[b_index] + eps

        fd_b = (loss_1_b - loss_2_b)/(2*eps)

        print("FD for b {0}".format(fd_b))
        print("Actual for b {0}".format(d[b_index][0]))
        relative_error_b = np.abs(fd_b - d[b_index][0]) /(np.abs(d[b_index][0]) + np.abs(fd_b))

        print("Relative error for b is {0}".format(relative_error_b))
        self.assertEqual( relative_error_b < 1e-5, True)

    def test_derivative_from_backpropagation_sigmoid_one_data_point(self):
        n_in_1 = 4
        n_out_1 = 5
        n_in_2 = 5
        n_out_2 = 10
        n_in_3 = 10
        n_out_3 = 6
        n_in_4 = 6
        n_out_4 = 4

        self.nn.cost_function = sl.Squared_Loss()
        self.nn.add_layer(n_in_1, n_out_1, sigmoid.Sigmoid())
        self.nn.add_layer(n_in_2, n_out_2, sigmoid.Sigmoid())
        self.nn.add_layer(n_in_3, n_out_3, sigmoid.Sigmoid())
        self.nn.add_layer(n_in_4, n_out_4, sigmoid.Sigmoid())
        X = np.random.random((n_in_1,1))
        C = self.nn.loss(X, X)
        eps = 0.000001

        y_hat = self.nn.feed_forward_and_backward(X, X)

        layer = np.random.randint(4)
        print("Layer is {0}".format(layer))

        w_index_row = np.random.randint(self.nn.layers[layer].W.shape[0])
        w_index_col = np.random.randint(self.nn.layers[layer].W.shape[1])
        b_index = np.random.randint(self.nn.layers[layer].b.shape[0])

        print("Element of W is ({0}, {1})".format(w_index_row, w_index_col))
        print("Element of b is {0}\n".format(b_index))
        # Get derivative of W
        dW, d = self.nn.get_derivative_W_and_delta(layer, y_hat, X, X)

        self.nn.layers[layer].W[w_index_row,w_index_col] = self.nn.layers[layer].W[w_index_row,w_index_col] + eps
        loss_1_W = self.nn.loss(X, X)
        self.nn.layers[layer].W[w_index_row,w_index_col] = self.nn.layers[layer].W[w_index_row,w_index_col] - 2*eps
        loss_2_W = self.nn.loss(X, X)

        fd_W = (loss_1_W - loss_2_W)/(2*eps)
        print("FD for W {0}".format(fd_W))
        print("Actual for W {0}".format(dW[w_index_row,w_index_col]))
        relative_error_W = np.abs(fd_W - dW[w_index_row, w_index_col])/(np.abs(dW[w_index_row, w_index_col]) + np.abs(fd_W))
        print("Relative error for W is {0}\n".format(relative_error_W))
        self.assertEqual( relative_error_W < 1e-5, True)
        #Need to reset this
        self.nn.layers[layer].W[w_index_row,w_index_col] = self.nn.layers[layer].W[w_index_row,w_index_col] + eps

        self.nn.layers[layer].b[b_index] = self.nn.layers[layer].b[b_index] + eps
        loss_1_b = self.nn.loss(X, X)
        self.nn.layers[layer].b[b_index] = self.nn.layers[layer].b[b_index] - 2*eps
        loss_2_b = self.nn.loss(X, X)
        self.nn.layers[layer].b[b_index] = self.nn.layers[layer].b[b_index] + eps

        fd_b = (loss_1_b - loss_2_b)/(2*eps)

        print("FD for b {0}".format(fd_b))
        print("Actual for b {0}".format(d[b_index][0]))
        relative_error_b = np.abs(fd_b - d[b_index][0]) /(np.abs(d[b_index][0]) + np.abs(fd_b))

        print("Relative error for b is {0}".format(relative_error_b))
        self.assertEqual( relative_error_b < 1e-5, True)

    def test_derivative_from_backpropagation_identity_multiple_data_points(self):
        n_in_1 = 4
        n_out_1 = 5
        n_in_2 = 5
        n_out_2 = 10
        n_in_3 = 10
        n_out_3 = 6
        n_in_4 = 6
        n_out_4 = 4

        self.nn.cost_function = sl.Squared_Loss()
        self.nn.add_layer(n_in_1, n_out_1, identity.Identity())
        self.nn.add_layer(n_in_2, n_out_2, identity.Identity())
        self.nn.add_layer(n_in_3, n_out_3, identity.Identity())
        self.nn.add_layer(n_in_4, n_out_4, identity.Identity())
        N=np.random.randint(1,25)
        X = np.random.random((n_in_1,N))
        print("Number of data points is {0}".format(N))
        C = self.nn.loss(X, X)
        eps = 1e-5

        y_hat = self.nn.feed_forward_and_backward(X, X)
        layer = np.random.randint(4)
        print("Layer is {0}".format(layer))

        w_index_row = np.random.randint(self.nn.layers[layer].W.shape[0])
        w_index_col = np.random.randint(self.nn.layers[layer].W.shape[1])
        b_index = np.random.randint(self.nn.layers[layer].b.shape[0])

        print("Element of W is ({0}, {1})".format(w_index_row, w_index_col))
        print("Element of b is {0}\n".format(b_index))
        # Get derivative of W

        dW, d = self.nn.get_derivative_W_and_delta_gd(layer, y_hat, X, X, N)

        self.nn.layers[layer].W[w_index_row,w_index_col] = self.nn.layers[layer].W[w_index_row,w_index_col] + eps
        loss_1_W = self.nn.loss(X, X)
        self.nn.layers[layer].W[w_index_row,w_index_col] = self.nn.layers[layer].W[w_index_row,w_index_col] - 2*eps
        loss_2_W = self.nn.loss(X, X)

        fd_W = (loss_1_W - loss_2_W)/(2*eps)
        print("FD for W {0}".format(fd_W))
        print("Actual for W {0}".format(dW[w_index_row,w_index_col]))
        relative_error_W = np.abs(fd_W - dW[w_index_row, w_index_col])/(np.abs(dW[w_index_row, w_index_col]) + np.abs(fd_W))
        print("Relative error for W is {0}\n".format(relative_error_W))
        self.assertEqual( relative_error_W < 1e-5, True)
        # Need to reset this
        self.nn.layers[layer].W[w_index_row,w_index_col] = self.nn.layers[layer].W[w_index_row,w_index_col] + eps

        self.nn.layers[layer].b[b_index] = self.nn.layers[layer].b[b_index] + eps
        loss_1_b = self.nn.loss(X, X)
        self.nn.layers[layer].b[b_index] = self.nn.layers[layer].b[b_index] - 2*eps
        loss_2_b = self.nn.loss(X, X)
        self.nn.layers[layer].b[b_index] = self.nn.layers[layer].b[b_index] + eps

        fd_b = (loss_1_b - loss_2_b)/(2*eps)

        print("FD for b {0}".format(fd_b))
        print("Actual for b {0}".format(d[b_index][0]))
        relative_error_b = np.abs(fd_b - d[b_index][0]) /(np.abs(d[b_index][0]) + np.abs(fd_b))

        print("Relative error for b is {0}".format(relative_error_b))
        self.assertEqual( relative_error_b < 1e-5, True)


    def test_derivative_from_backpropagation_sigmoid_multiple_data_points(self):
        n_in_1 = 4
        n_out_1 = 5
        n_in_2 = 5
        n_out_2 = 10
        n_in_3 = 10
        n_out_3 = 6
        n_in_4 = 6
        n_out_4 = 4


        self.nn.add_layer(n_in_1, n_out_1, sigmoid.Sigmoid())
        self.nn.add_layer(n_in_2, n_out_2, sigmoid.Sigmoid())
        self.nn.add_layer(n_in_3, n_out_3, sigmoid.Sigmoid())
        self.nn.add_layer(n_in_4, n_out_4, sigmoid.Sigmoid())
        N=np.random.randint(1,25)
        X = np.random.random((n_in_1,N))
        print("Number of data points is {0}".format(N))
        C = self.nn.loss(X, X)
        eps = 1e-5

        y_hat = self.nn.feed_forward_and_backward(X, X)
        layer = np.random.randint(4)
        print("Layer is {0}".format(layer))

        w_index_row = np.random.randint(self.nn.layers[layer].W.shape[0])
        w_index_col = np.random.randint(self.nn.layers[layer].W.shape[1])
        b_index = np.random.randint(self.nn.layers[layer].b.shape[0])

        print("Element of W is ({0}, {1})".format(w_index_row, w_index_col))
        print("Element of b is {0}\n".format(b_index))
        # Get derivative of W

        dW, d = self.nn.get_derivative_W_and_delta_gd(layer, y_hat, X, X, N)

        self.nn.layers[layer].W[w_index_row,w_index_col] = self.nn.layers[layer].W[w_index_row,w_index_col] + eps
        loss_1_W = self.nn.loss(X, X)
        self.nn.layers[layer].W[w_index_row,w_index_col] = self.nn.layers[layer].W[w_index_row,w_index_col] - 2*eps
        loss_2_W = self.nn.loss(X, X)

        fd_W = (loss_1_W - loss_2_W)/(2*eps)
        print("FD for W {0}".format(fd_W))
        print("Actual for W {0}".format(dW[w_index_row,w_index_col]))
        relative_error_W = np.abs(fd_W - dW[w_index_row, w_index_col])/(np.abs(dW[w_index_row, w_index_col]) + np.abs(fd_W))
        print("Relative error for W is {0}\n".format(relative_error_W))
        self.assertEqual( relative_error_W < 1e-5, True)
        # Need to reset this
        self.nn.layers[layer].W[w_index_row,w_index_col] = self.nn.layers[layer].W[w_index_row,w_index_col] + eps

        self.nn.layers[layer].b[b_index] = self.nn.layers[layer].b[b_index] + eps
        loss_1_b = self.nn.loss(X, X)
        self.nn.layers[layer].b[b_index] = self.nn.layers[layer].b[b_index] - 2*eps
        loss_2_b = self.nn.loss(X, X)
        self.nn.layers[layer].b[b_index] = self.nn.layers[layer].b[b_index] + eps

        fd_b = (loss_1_b - loss_2_b)/(2*eps)

        print("FD for b {0}".format(fd_b))
        print("Actual for b {0}".format(d[b_index][0]))
        relative_error_b = np.abs(fd_b - d[b_index][0]) /(np.abs(d[b_index][0]) + np.abs(fd_b))

        print("Relative error for b is {0}".format(relative_error_b))
        self.assertEqual( relative_error_b < 1e-5, True)

    def test_derivative_from_backpropagation_sigmoid_log_loss_multiple_data_points(self):
        n_in_1 = 4
        n_out_1 = 5
        n_in_2 = 5
        n_out_2 = 10
        n_in_3 = 10
        n_out_3 = 6
        n_in_4 = 6
        n_out_4 = 4

        self.nn.cost_function = ll.Logistic_Loss()
        self.nn.add_layer(n_in_1, n_out_1, sigmoid.Sigmoid())
        self.nn.add_layer(n_in_2, n_out_2, sigmoid.Sigmoid())
        self.nn.add_layer(n_in_3, n_out_3, sigmoid.Sigmoid())
        self.nn.add_layer(n_in_4, n_out_4, sigmoid.Sigmoid())
        N=np.random.randint(1,25)
        X = np.random.random((n_in_1, N))
        Y = np.random.randint(3, size=(N, ))

        print("Number of data points is {0}".format(N))
        self.nn.cost_function = ll.Logistic_Loss()
        C = self.nn.loss(X, Y)
        eps = 1e-5

        y_hat = self.nn.feed_forward_and_backward(X, Y)
        layer = np.random.randint(4)
        print("Layer is {0}".format(layer))

        w_index_row = np.random.randint(self.nn.layers[layer].W.shape[0])
        w_index_col = np.random.randint(self.nn.layers[layer].W.shape[1])
        b_index = np.random.randint(self.nn.layers[layer].b.shape[0])

        print("Element of W is ({0}, {1})".format(w_index_row, w_index_col))
        print("Element of b is {0}\n".format(b_index))
        # Get derivative of W

        dW, d = self.nn.get_derivative_W_and_delta_gd(layer, y_hat, Y, X, N)

        self.nn.layers[layer].W[w_index_row,w_index_col] = self.nn.layers[layer].W[w_index_row,w_index_col] + eps
        loss_1_W = self.nn.loss(X, Y)
        self.nn.layers[layer].W[w_index_row,w_index_col] = self.nn.layers[layer].W[w_index_row,w_index_col] - 2*eps
        loss_2_W = self.nn.loss(X, Y)

        fd_W = (loss_1_W - loss_2_W)/(2*eps)
        print("FD for W {0}".format(fd_W))
        print("Actual for W {0}".format(dW[w_index_row,w_index_col]))
        relative_error_W = np.abs(fd_W - dW[w_index_row, w_index_col])/(np.abs(dW[w_index_row, w_index_col]) + np.abs(fd_W))
        print("Relative error for W is {0}\n".format(relative_error_W))
        self.assertEqual( relative_error_W < 1e-5, True)
        # Need to reset this
        self.nn.layers[layer].W[w_index_row,w_index_col] = self.nn.layers[layer].W[w_index_row,w_index_col] + eps

        self.nn.layers[layer].b[b_index] = self.nn.layers[layer].b[b_index] + eps
        loss_1_b = self.nn.loss(X, Y)
        self.nn.layers[layer].b[b_index] = self.nn.layers[layer].b[b_index] - 2*eps
        loss_2_b = self.nn.loss(X, Y)
        self.nn.layers[layer].b[b_index] = self.nn.layers[layer].b[b_index] + eps

        fd_b = (loss_1_b - loss_2_b)/(2*eps)

        print("FD for b {0}".format(fd_b))
        print("Actual for b {0}".format(d[b_index][0]))
        relative_error_b = np.abs(fd_b - d[b_index][0]) /(np.abs(d[b_index][0]) + np.abs(fd_b))

        print("Relative error for b is {0}".format(relative_error_b))
        self.assertEqual( relative_error_b < 1e-5, True)

    def test_derivative_from_backpropagation_sigmoid_cross_entropy_loss_multiple_data_points(self):
        n_in_1 = 4
        n_out_1 = 5
        n_in_2 = 5
        n_out_2 = 10
        n_in_3 = 10
        n_out_3 = 6
        n_in_4 = 6
        n_out_4 = 4

        self.nn.cost_function = cel.Cross_Entropy_Loss()
        self.nn.add_layer(n_in_1, n_out_1, sigmoid.Sigmoid())
        self.nn.add_layer(n_in_2, n_out_2, sigmoid.Sigmoid())
        self.nn.add_layer(n_in_3, n_out_3, sigmoid.Sigmoid())
        self.nn.add_layer(n_in_4, n_out_4, sigmoid.Sigmoid())
        N=np.random.randint(1,25)
        X = np.random.random((n_in_1, N))
        Y = np.random.randint(3, size=(N, ))

        print("Number of data points is {0}".format(N))
        self.nn.cost_function = ll.Logistic_Loss()
        C = self.nn.loss(X, Y)
        eps = 1e-5

        y_hat = self.nn.feed_forward_and_backward(X, Y)
        layer = np.random.randint(4)
        print("Layer is {0}".format(layer))

        w_index_row = np.random.randint(self.nn.layers[layer].W.shape[0])
        w_index_col = np.random.randint(self.nn.layers[layer].W.shape[1])
        b_index = np.random.randint(self.nn.layers[layer].b.shape[0])

        print("Element of W is ({0}, {1})".format(w_index_row, w_index_col))
        print("Element of b is {0}\n".format(b_index))
        # Get derivative of W

        dW, d = self.nn.get_derivative_W_and_delta_gd(layer, y_hat, Y, X, N)

        self.nn.layers[layer].W[w_index_row,w_index_col] = self.nn.layers[layer].W[w_index_row,w_index_col] + eps
        loss_1_W = self.nn.loss(X, Y)
        self.nn.layers[layer].W[w_index_row,w_index_col] = self.nn.layers[layer].W[w_index_row,w_index_col] - 2*eps
        loss_2_W = self.nn.loss(X, Y)

        fd_W = (loss_1_W - loss_2_W)/(2*eps)
        print("FD for W {0}".format(fd_W))
        print("Actual for W {0}".format(dW[w_index_row,w_index_col]))
        relative_error_W = np.abs(fd_W - dW[w_index_row, w_index_col])/(np.abs(dW[w_index_row, w_index_col]) + np.abs(fd_W))
        print("Relative error for W is {0}\n".format(relative_error_W))
        self.assertEqual( relative_error_W < 1e-5, True)
        # Need to reset this
        self.nn.layers[layer].W[w_index_row,w_index_col] = self.nn.layers[layer].W[w_index_row,w_index_col] + eps

        self.nn.layers[layer].b[b_index] = self.nn.layers[layer].b[b_index] + eps
        loss_1_b = self.nn.loss(X, Y)
        self.nn.layers[layer].b[b_index] = self.nn.layers[layer].b[b_index] - 2*eps
        loss_2_b = self.nn.loss(X, Y)
        self.nn.layers[layer].b[b_index] = self.nn.layers[layer].b[b_index] + eps

        fd_b = (loss_1_b - loss_2_b)/(2*eps)

        print("FD for b {0}".format(fd_b))
        print("Actual for b {0}".format(d[b_index][0]))
        relative_error_b = np.abs(fd_b - d[b_index][0]) /(np.abs(d[b_index][0]) + np.abs(fd_b))

        print("Relative error for b is {0}".format(relative_error_b))
        self.assertEqual( relative_error_b < 1e-5, True)

    def test_derivative_from_backpropagation_if_W_is_identity_matrix_and_b_is_zero(self):
        n_in_1 = 4
        n_out_1 = 4
        n_in_2 = 4
        n_out_2 = 4
        n_in_3 = 4
        n_out_3 = 4
        n_in_4 = 4
        n_out_4 = 4


        self.nn.add_layer(n_in_1, n_out_1, identity.Identity())
        self.nn.add_layer(n_in_2, n_out_2, identity.Identity())
        self.nn.add_layer(n_in_3, n_out_3, identity.Identity())
        self.nn.add_layer(n_in_4, n_out_4, identity.Identity())
        for layer in self.nn.layers:
            layer.W = np.eye(4)
            layer.b = np.zeros((4,1))

        N=np.random.randint(1,25)
        X = np.random.random((n_in_1,N))
        print("Number of data points is {0}".format(N))
        C = self.nn.loss(X, X)
        eps = 1e-5

        y_hat = self.nn.feed_forward_and_backward(X, X)
        layer = np.random.randint(4)
        print("Layer is {0}".format(layer))

        w_index_row = np.random.randint(self.nn.layers[layer].W.shape[0])
        w_index_col = np.random.randint(self.nn.layers[layer].W.shape[1])
        b_index = np.random.randint(self.nn.layers[layer].b.shape[0])

        print("Element of W is ({0}, {1})".format(w_index_row, w_index_col))
        print("Element of b is {0}\n".format(b_index))
        # Get derivative of W

        dW, d = self.nn.get_derivative_W_and_delta_gd(layer, y_hat, X, X, N)


        print("Actual for W {0}".format(dW[w_index_row,w_index_col]))
        if w_index_col == w_index_row:
            to_test = (dW[w_index_row, w_index_col] == self.nn.l)
        else:
            to_test = dW[w_index_row, w_index_col] == 0.0
        self.assertEqual(to_test, True)
        # Need to reset this

        print("Actual for b {0}".format(d[b_index][0]))

        self.assertEqual( d[b_index][0] == 0, True)

    #######################
    ## Function Tests
    #######################

    def test_cost_function_square_loss(self):
        cf = sl.Squared_Loss()
        Y = np.array([[1,0,1], [0,2,0]]).T
        X = np.array([[0.5, 0.1, 0.5], [0,1,0]]).T
        self.assertEqual(cf.loss(X, Y), 0.3775)
        self.assertEqual(cf.derivative(X, Y).tolist(), np.array([[-0.25], [-0.45], [-0.25]]).tolist())

    def test_squared_loss(self):
        sqrd_loss = sl.Squared_Loss()
        X = np.array([1,2,3,4])
        Y_hat = np.array([1.1,2.1,3.3,4.2])
        print(sqrd_loss.loss(Y_hat, X))

    def test_squared_loss_two_inputs(self):
        sqrd_loss = sl.Squared_Loss()
        Y = np.array([[1,2,3,4], [5,6,7,8]]).T
        Y_hat = np.array([[1,2,3,6], [5,6,7,9]]).T
        print(sqrd_loss.loss(Y_hat, Y))

    def test_squared_loss_derivative_two_inputs(self):
        sqrd_loss = sl.Squared_Loss()
        Y = np.array([[1,2,3,4], [5,6,7,8]]).T
        Y_hat = np.array([[3,2,3,4], [5,6,7,9]]).T
        print(sqrd_loss.derivative(Y_hat, Y))

    def test_logistic_loss_two_inputs_small_loss(self):
        log_loss = ll.Logistic_Loss()
        Y = np.array([1,0,3])
        Y_hat = np.array([[-0.0000, 9, -10, 0.00001], [8, 0.01, 0.01, 0.01], [0.01, 0.01, 0.01, 22]]).T
        print(log_loss.loss(Y_hat, Y))

    def test_logistic_loss_finite_difference_derivative_per_data_point(self):
        log_loss = ll.Logistic_Loss()
        Y = np.array([2, 0, 3])
        Y_hat = np.array([[10, 0.01, 0.01, 0.01], [0.97, 0.01, 0.01, 0.01], [0.01, 0.01, 0.01, 0.97]]).T

        index = np.random.randint(4)
        print("index is {0}".format(index))

        eps = 0.000001
        eps_vec = np.zeros((4,1))
        eps_vec[index] = eps
        fd = (log_loss.loss(Y_hat + eps_vec, Y) - log_loss.loss(Y_hat - eps_vec, Y))/(2*eps)
        d = log_loss.derivative(Y_hat, Y)[index]
        print(d)
        print(fd)
        print("FD {0}".format(fd))
        print("Actual {0}".format(d))
        if np.abs(fd) < 1e-16 and np.abs(d) < 1e-16:
            relative_error = 0
        else:
            relative_error = np.abs(fd - d)/(np.abs(d) + np.abs(fd))
        print("Relative error is {0}\n".format(relative_error))
        self.assertEqual( relative_error < 1e-8, True)

        print(log_loss.loss(Y_hat, Y))

    def test_logistic_loss_overflow(self):
        log_loss = ll.Logistic_Loss()
        Y = np.array([2, 0, 3])
        Y_hat = np.array([[10000, 0.01, 0.01, 0.01], [0.97, 0.01, 0.01, 0.01], [0.01, 0.01, 0.01, 0.97]]).T

        index = np.random.randint(4)
        print("index is {0}".format(index))

        eps = 0.000001
        eps_vec = np.zeros((4,1))
        eps_vec[index] = eps
        fd = (log_loss.loss(Y_hat + eps_vec, Y) - log_loss.loss(Y_hat - eps_vec, Y))/(2*eps)
        d = log_loss.derivative(Y_hat, Y)[index]
        print(d)
        print(fd)
        print("FD {0}".format(fd))
        print("Actual {0}".format(d))
        if np.abs(fd) < 1e-16 and np.abs(d) < 1e-16:
            relative_error = 0
        else:
            relative_error = np.abs(fd - d)/(np.abs(d) + np.abs(fd))
        print("Relative error is {0}\n".format(relative_error))
        self.assertEqual( relative_error < 1e-5, True)

        print(log_loss.loss(Y_hat, Y))

    def test_finite_difference_on_sigmoid(self):
        sig = sigmoid.Sigmoid()
        X = np.random.random((4,))
        index = np.random.randint(4)
        print("index is {0}".format(index))

        eps = 0.0001
        eps_vec = np.zeros((4,))
        eps_vec[index] = eps
        fd = (sig.eval(X + eps_vec)[index] - sig.eval(X - eps_vec)[index])/(2*eps)
        print("FD {0}".format(fd))
        d = sig.derivative(X)[index]

        print("Actual {0}".format(d))

        relative_error = np.abs(fd - d)/(np.abs(d) + np.abs(fd))
        print("Relative error for W is {0}\n".format(relative_error))
        self.assertEqual(relative_error < 1e-8, True)

    def test_regularised_square_loss(self):
        n_in_1 = 4
        n_out_1 = 5
        n_in_2 = 5
        n_out_2 = 6
        n_in_3 = 6
        n_out_3 = 4

        self.nn.add_layer(n_in_1, n_out_1, sigmoid.Sigmoid())
        self.nn.add_layer(n_in_2, n_out_2, sigmoid.Sigmoid())
        self.nn.add_layer(n_in_3, n_out_3, identity.Identity())
        X = np.random.random((4, 10))
        print(self.nn.loss(X, X))

