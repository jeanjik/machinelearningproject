import random
import numpy as np
import functions as fn
import loader as ld


class Network(object):
    def __init__(self, sizes, activation, error):
        # passa in input il num di neuroni di ogni layer: array[input, L1, L2, ...]
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]  # ARRAY DI: matrice riga 1xL
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        self.activation_function = activation
        self.error_function = error
        self.dictionary_function = fn.dictionary()

    def forward_propagation(self, a):
        for b, w, i in zip(self.biases, self.weights, range(0,self.num_layers)):
            f = self.dictionary_function[self.activation_function[i]][0]
            a = f(np.dot(w, a) + b)
        output_function = self.dictionary_function[self.activation_function[-1]][0]
        return output_function(a)

    def back_propagation(self, input, target):
        delta_err_b = [np.zeros(b.shape) for b in self.biases]  # creiamo la matrice delta errore
        derivative_err_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = input
        activations = [input]
        zs = []
        for b, w, i in zip(self.biases, self.weights, range(0, self.num_layers)):
            z = np.dot(w, activation) + b
            zs.append(z)
            f = self.dictionary_function[self.activation_function[i]][0]
            activation = f(z)
            activations.append(activation)

        # passaggio backward
        derivative_error_function = self.dictionary_function[self.error_function][1]
        delta_error = derivative_error_function(activations[-1], target)
        derivative_activation_function = self.dictionary_function[self.activation_function[-1]][1]
        act = derivative_activation_function(zs[-1])
        delta_err_b[-1] = delta_error * act
        derivative_err_w[-1] = np.dot(delta_error, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            derivative_activation_function = self.dictionary_function[self.activation_function[-l]][1]
            sp = derivative_activation_function(z)
            delta_error = np.dot(self.weights[-l + 1].transpose(), delta_error) * sp
            delta_err_b[-l] = delta_error
            derivative_err_w[-l] = np.dot(delta_error, activations[-l - 1].transpose())
        return delta_err_b, derivative_err_w

    def batch_gradient_descent(self, training_data, epochs, eta, momentum, test_data, k=None):
        # training_data e' una lista di tuple (x, y) dove x e' l'input e y e' la corrispondente label
        pre_w = [np.zeros(w.shape) for w in self.weights]  # per l'iterazione precedente
        pre_b = [np.zeros(b.shape) for b in self.biases]
        err = []
        for j in range(epochs):
            random.shuffle(training_data)
            sum_w = [np.zeros(w.shape) for w in self.weights]
            sum_b = [np.zeros(b.shape) for b in self.biases]
            for input, label in training_data:
                der_b, der_w = self.back_propagation(input, label)
                sum_b = [sb + derb for sb, derb in zip(sum_b, der_b)]
                sum_w = [sw + derw for sw, derw in zip(sum_w, der_w)]
            self.weights = [w - (eta * wl + momentum * pwl) for w, wl, pwl in zip(self.weights, sum_w, pre_w)]
            self.biases = [b - (eta * bl + momentum * pbl) for b, bl, pbl in zip(self.biases, sum_b, pre_b)]
            pre_b = pre_b + sum_b
            pre_w = pre_w + sum_w
            print("EPOCA " + str(j+1) + " NE HO INCARRATE " + str(self.evaluate(test_data)) + " SU " + str(len(test_data)))
            error_function = self.dictionary_function[self.error_function][0]
            err.append(self.calculate_error(test_data, error_function))
            print("ERRORE EPOCA " + str(j+1) + str(err[j]) + " SU " + str(len(test_data)))
            if k is not None:
                stop = self.check_early_stopping(err, j, k)
                if stop is True:
                    return err
        return err

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.forward_propagation(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def check_early_stopping(self, err, epoch, k):
        if epoch - k > 0:
            if err[epoch] > err[epoch - k]:
                return True
        return False

    def calculate_error(self, test_data, error_function):
        err = 0
        for inp, target in test_data:
            out = self.forward_propagation(inp)
            err = err + error_function(out, ld.vectorized_result(target))
        return err


def main():
    error = "least_square"
    activation = ["sigmoid", "sigmoid","identity"]
    val = Network([784, 100, 10], activation, error)
    tr_d = ld.load_data(2000)
    t_data = ld.load_test_data(100)
    print("INIZIO TRAINING")
    val.batch_gradient_descent(tr_d, 100, 0.002, 0.85, t_data)


main()
