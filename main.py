import random
import numpy as np
import functions as fn


class Network(object):
    def __init__(self, sizes):
        # passa in input il num di neuroni di ogni layer: array[input, L1, L2, ...]
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(1, y) for y in sizes[1:]]  # ARRAY DI: matrice riga 1xL
        self.weights = [np.random.randn(x, y) for x, y in zip(sizes[1:], sizes[:-1])]
        # ARRAY Per ogni layer L: LxL-1 (quando L=1 rappresentiamo la size dell'input)

    def forward_propagation(self, input, activation_function=fn.sigmoid, output_function=fn.identity):
        # activation sono i valori di input per i nodi (per il primo strato sono proprio gli input x)
        activation = input
        activation_array = [input.T]
        for weights, biases, layer in zip(self.weights, self.biases, range(0, self.num_layers)):
            # weights: LxL-1 , activation: L-1x1 , w * a : Lx1 , biases.T : Lx1
            a = np.dot(weights, activation) + biases.transpose()
            # activation_array[layer][:] = a.T
            activation_array.append(a.T)
            activation = activation_function(a)  # activation: Lx1 (in questo caso e' cio' che noi chiamiamo z)
        return output_function(activation), activation_array

    def back_propagation(self, activate, output, target,
                         activation_function=fn.sigmoid, derivative_activation_function=fn.derivative_sigmoid,
                         derivative_cost_function=fn.derivative_least_square):
        delta_err_b = [np.zeros(b.shape) for b in self.biases]  # creiamo la matrice delta errore
        derivative_err_w = [np.zeros(w.shape) for w in self.weights]
        error = derivative_cost_function(output=output, target=target)  # nodi output
        delta_err_b[-1] = error.T * derivative_activation_function(activate[-1])  # caso base formula
        derivative_err_w[-1] = np.dot(delta_err_b[-1].T, activation_function(activate[-2]))
        for layer in xrange(2, self.num_layers):  # nodi interni
            delta_err_b[-layer] = np.dot(delta_err_b[-layer + 1], self.weights[-layer + 1]) * activation_function(
                activate[-layer])  # formula ricorrente per i nodi interni
            derivative_err_w[-layer] = np.dot(delta_err_b[-layer].T, activation_function(activate[-layer - 1]))
        return delta_err_b, derivative_err_w

    def batch_gradient_descent(self, training_data, epochs, eta, momentum, test_data=None):
        # training_data e' una lista di tuple (x, y) dove x e' l'input e y e' la corrispondente label
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            sum_w = [np.zeros(w.shape) for w in self.weights]
            sum_b = [np.zeros(b.shape) for b in self.biases]
            for input, target in training_data:
                output, activation = self.forward_propagation(input)
                der_w, der_b = self.back_propagation(activation, output, target)
                sum_b = [sb + derb for sb, derb in zip(sum_b, der_b)]
                sum_w = [sw + derw for sw, derw in zip(sum_w, der_w)]
            self.weights = [(w - eta/len(training_data)) * momentum * wl for w, wl in zip(self.weights, sum_w)]
            self.biases = [(b - eta / len(training_data)) * momentum * bl for b, bl in zip(self.biases, sum_b)]


    def evaluate_error(self, input, target, error_function):
        output, _ = self.forward_propagation(input)
        print(error_function(output, target))



def main():
    val = Network([4, 3, 2, 2])
    print('----PESI---')
    print(val.weights)
    print('----BIASES---')
    print(val.biases)
    input = np.random.randn(4, 1)
    print('----INPUT---')
    print(input)
    print('----APPLICO FORWARD PROPAGATION---')
    output, activation = val.forward_propagation(input)
    print('----OUTPUT---')
    print(output)
    print('---ACTIVATION---')
    print(activation)

    expected = np.array([[0], [1]])
    print('----TARGET---')
    print(expected)
    print('----APPLICO BACK PROPAGATION---')
    delta, der_err = val.back_propagation(activate=activation, output=output, target=expected)
    print 'DELTA:'
    print delta
    print 'DERIVATE ERRORE: '
    print der_err


main()
