import random
import numpy as np


class Network(object):
    def __init__(self, sizes):
        #passa in input il num di neuroni di ogni layer: array[input, L1, L2, ...]
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(1, y) for y in sizes[1:]]  # ARRAY DI: matrice riga 1xL
        self.weights = [np.random.randn(x, y) for x, y in zip(sizes[1:], sizes[:-1])]
        #ARRAY Per ogni layer L: LxL-1 (quando L=1 rappresentiamo la size dell'input)

    def forward_propagation(self, input, activation_function):
        # activation sono i valori di input per i nodi (per il primo strato sono proprio gli input x)
        activation = input
        activation_array = [input.T]
        for weights, biases, layer in zip(self.weights, self.biases, range(0, self.num_layers)):
            # weights: LxL-1 , activation: L-1x1 , w * a : Lx1 , biases.T : Lx1
            a = np.dot(weights, activation) + biases.transpose()
            # activation_array[layer][:] = a.T
            activation_array.append(a.T)
            activation = activation_function(a) # activation: Lx1 (in questo caso e' cio' che noi chiamiamo z

        return activation, activation_array

    def back_propagation(self, activate, output, target, activation_function, derivative_activation_function,
                         derivative_cost_function):
        delta_err_b = [np.zeros(b.shape) for b in self.biases]  # creiamo la matrice delta errore
        derivative_err_w = [np.zeros(w.shape) for w in self.weights]
        error = derivative_cost_function(output=output, target=target) # nodi output
        delta_err_b[-1] = error.T * derivative_activation_function(activate[-1]) # caso base formula
        derivative_err_w[-1] = np.dot(delta_err_b[-1].T, activation_function(activate[-2]))
        for l in xrange(2, self.num_layers): # nodi interni
            delta_err_b[-l] = np.dot(delta_err_b[-l+1], self.weights[-l+1]) * activation_function(activate[-l]) # formula ricorrente per i nodi interni
            derivative_err_w[-l] = np.dot(delta_err_b[-l].T, activation_function(activate[-l-1]))
        return delta_err_b, derivative_err_w


    def stochastic_gradient_descent(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        # training_data e' una lista di tuple (x, y) dove x e' l'input e y e' la corrispondente label
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in xrange(0, n, mini_batch_size)
                # ritorna un range che parte da 0 e arriva a n, con step mini_batch_size
            ]
            for mini_batch in mini_batches:
                print("cazzo")


    def update_mini_batches(self, mini_batch,eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases] # shape ritorna le dimensioni della matrice
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = "cazzo"  # TODO: qua ci va la backpropagation
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]


def derivative_least_square(output, target):
    return output - target


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def identity(z):
    return z


def derivative_sigmoid(z):
    return z * (1 - z)


val = Network([4, 3, 2])
print('----PESI---')
print(val.weights)
print('----BIASES---')
print(val.biases)
input = np.random.randn(4, 1)
print('----INPUT---')
print(input)
print('----APPLICO FORWARD PROPAGATION---')
output,activation=val.forward_propagation(input,sigmoid)
print('----OUTPUT---')
print(output)
print('---ACTIVATION---')
print(activation)

expected = np.array([[0], [1]])
print('----TARGET---')
print(expected)
print('----APPLICO BACK PROPAGATION---')
delta, der_err = val.back_propagation(activate=activation, output=output, target=expected, activation_function=sigmoid,
                             derivative_activation_function=derivative_sigmoid, derivative_cost_function=derivative_least_square)
print 'DELTA:'
print delta
print 'DERIVATE ERRORE: '
print der_err





