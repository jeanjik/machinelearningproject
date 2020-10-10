import random
import numpy as np
import functions as fn
import loader as ld

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
            dot = np.dot(weights, activation)
            a = np.array([d + b for d, b in zip(dot, biases.T)])
            activation_array.append(a)
            activation = activation_function(a)  # activation: Lx1 (in questo caso e' cio' che noi chiamiamo z)
        return output_function(activation), activation_array

    def back_propagation(self, activate, output, target,
                         activation_function=fn.sigmoid, derivative_activation_function=fn.derivative_sigmoid,
                         derivative_cost_function=fn.derivative_least_square):
        delta_err_b = [np.zeros(b.shape) for b in self.biases]  # creiamo la matrice delta errore
        derivative_err_w = [np.zeros(w.shape) for w in self.weights]
        der_error = derivative_cost_function(output=output, target=target)[np.newaxis]
        act = derivative_activation_function(activate[-1])
        delta_err_b[-1] = np.array(der_error * act.T)  # nodi output
        tmp=activation_function(activate[-2])[np.newaxis]
        derivative_err_w[-1] = np.dot(delta_err_b[-1].T, tmp)
        for layer in range(2, self.num_layers):  # nodi interni
            temp = np.dot(delta_err_b[-layer + 1], self.weights[-layer + 1])
            temp2 = activation_function(activate[-layer]).T
            delta_err_b[-layer] = temp * temp2  # formula ricorrente per i nodi interni
            temp3 = activation_function(activate[-layer - 1])[np.newaxis]
            derivative_err_w[-layer] = np.dot(delta_err_b[-layer].T, temp3)
        return delta_err_b, derivative_err_w

    def batch_gradient_descent(self, training_data, epochs, eta, momentum):
        # training_data e' una lista di tuple (x, y) dove x e' l'input e y e' la corrispondente label
        err = np.zeros(epochs)
        for j in range(epochs):
            random.shuffle(training_data)
            sum_w = [np.zeros(w.shape) for w in self.weights]
            sum_b = [np.zeros(b.shape) for b in self.biases]
            for input, label in training_data:
                output, activation = self.forward_propagation(input)
                der_b, der_w = self.back_propagation(activation, output, label)
                sum_b = [sb + derb for sb, derb in zip(sum_b, der_b)]
                sum_w = [sw + derw for sw, derw in zip(sum_w, der_w)]
            self.weights = [(w - eta/len(training_data)) * momentum * wl for w, wl in zip(self.weights, sum_w)]
            self.biases = [(b - eta / len(training_data)) * momentum * bl for b, bl in zip(self.biases, sum_b)]
            err[j] = self.evaluate_error(training_data, fn.least_square)
            print(err[j])
        return err

    def evaluate_error(self, training_data, error_function):
        err = 0
        for input, target in training_data:
            output, _ = self.forward_propagation(input)
            err = error_function(output, target)
        return err/len(training_data)

    def evaluate(self, test_data):
        test_results = []
        for x, y in test_data:
            out, _ = self.forward_propagation(x)
            res = (out.T, y)
            test_results.append(res)
        print(test_results)


def main():
    val = Network([784, 200, 10])
    input = np.random.rand(20, 784)
    target = np.random.rand(20, 10)
    training_data = [(i, t) for i, t in zip(input, target)]
    tr_d = ld.load_data(0, 1000)
    test_data = ld.load_data(0, 200)
    # print('----INPUT---')
    # print(input)
    # print('----APPLICO FORWARD PROPAGATION---')
    # output, activation = val.forward_propagation(input)
    # print('----OUTPUT---')
    # print(output)
    # print('---ACTIVATION---')
    # print(activation)
    #
    # expected = target
    # print('----TARGET---')
    # print(expected)
    # print('----APPLICO BACK PROPAGATION---')
    # delta, der_err = val.back_propagation(activate=activation, output=output, target=expected)
    # print 'DELTA:'
    # print delta
    # print 'DERIVATE ERRORE: '
    # print der_err
    err = val.batch_gradient_descent(tr_d, 5, 2, 0.9)
    print(err)
    print(val.evaluate(test_data))


main()
