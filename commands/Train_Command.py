__author__ = 's1463104'

import copy
from optimisation.results import Results
import cPickle

class Train_Command(object):

    def __init__(self, algorithm_name, step_size, epochs, minibatch_size=0, inner_it_multiplier=5, mu=0):
        self.algorithm_name = algorithm_name
        self.step_size = step_size
        self.minibatch_size = minibatch_size
        self.number_epochs = epochs
        self.inner_it_multiplier = inner_it_multiplier
        self.mu = mu
        self.nn = None


    def set_nn(self, nn):
        self.nn = copy.deepcopy(nn)

    def execute(self, X, Y, X_test, Y_test, path_to_pickle):
        Y_hat = self.nn.feed_forward(X_test)
        accuracy_initial = self.nn.classification_accuracy_given_output(Y_hat, Y_test)
        test_data = [X_test, Y_test]
        N = X.shape[1]
        max_it = self.number_epochs*N

        if self.algorithm_name == "gd":
            print("=== Gradient Descent ===")
            costs, accuracies = self.nn.train_gd(X, Y, max_it, step_size=self.step_size, test_data=test_data)
        elif self.algorithm_name == "sgd":
            print("=== SGD ===")
            costs, accuracies = self.nn.train_sgd(X, Y, max_it, step_size=self.step_size, mini_batch_size=self.minibatch_size,
                                                test_data=test_data)
        elif self.algorithm_name == "saga":
            print("=== SAGA ===")
            costs, accuracies = self.nn.train_saga(X, Y, max_it, step_size=self.step_size, test_data=test_data)
        elif self.algorithm_name == "svrg":
            print("=== SVRG ===")
            costs, accuracies = self.nn.train_svrg(X, Y, max_it, step_size=self.step_size, test_data=test_data, inner_it_multiplier=self.inner_it_multiplier)
        elif self.algorithm_name == "sdca":
            print("=== SDCA ===")
            costs, accuracies = self.nn.train_sdca(X, Y, max_it, step_size=self.step_size, test_data=test_data)

        print("\n=== lambda={0}, N={1}, step size={2}, number of epochs={3}, minibatch size={4}, inner_it_multiplier={5}, momentum={6} ===\n".format(self.nn.l, N,
                                                                                                      self.step_size,
                                                                                                      self.number_epochs,
                                                                                                      self.minibatch_size,
                                                                                                      self.inner_it_multiplier,
                                                                                                      self.mu))

        print("    Initial accuracy on test data is {0}".format(accuracy_initial))
        Y_hat = self.nn.feed_forward(X_test)
        accuracy_final = self.nn.classification_accuracy_given_output(Y_hat, Y_test)
        results = Results.Results(N, max_it, self.step_size, costs, accuracies, algorithm_used=self.algorithm_name,
                              test_accuracy_final=accuracy_final, test_accuracy_initial=accuracy_initial)
        file_name = path_to_pickle + \
                    "{0}_lambda_{1}_step_size_{2}_number_data_{3}_number_epochs_{4}_minibatch_size_{5}.pickle"\
                        .format(self.algorithm_name,
                        self.nn.l,
                        self.step_size,
                        N,
                        self.number_epochs,
                        self.minibatch_size)

        f = file(file_name, "w")
        cPickle.dump(results, file=f)
        f.close()

        print(costs)
        print(accuracies)
        print("    Final accuracy on test data is {0}".format(accuracy_final))

    def __str__(self):
        return  "    Train command - Algorithm: {0} - Step Size: {1} - Minibatch Size: {2} - Number of Epochs: {3} - Number of inner multiplier: {4}".format(
            self.algorithm_name, self.step_size, self.minibatch_size, self.number_epochs, self.inner_it_multiplier
        )