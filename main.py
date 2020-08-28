import random
import numpy as np


class Network(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        # randn ritorna una matrice di dimensione y, 1 - ovvero una matrice colonna il cui numero di righe e' pari
        # al numero di nodi nello strato corrente (escluso lo strato di input)
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        # i biases e weights sono salvati come liste di matrici per cui - ad esempio - weights[1] sara' una matrice
        # contenente i pesi che connettono il secondo e terzo strato di neuroni

    def feedforward(self, activation, activation_function):
        for biases, weights in zip(self.biases, self.weights):
            activation = activation_function(np.dot(weights, activation) + biases)
            # np.dot e' una funzione che effettua la moltiplicazione matriciale


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))
