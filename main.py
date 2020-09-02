import random
import numpy as np


class Network(object):
    def __init__(self, sizes):
        #passa in input il num di neuroni di ogni layer: array[input, L1, L2, ...]
        self.num_layers = len(sizes)
        self.sizes = sizes
        #self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.biases = [np.random.randn(1, y) for y in sizes[1:]] #ARRAY DI: matrice riga 1xL
        # randn ritorna una matrice di dimensione y, 1 - ovvero una matrice colonna il cui numero di righe e' pari
        # al numero di nodi nello strato corrente (escluso lo strato di input)

        self.weights = [np.random.randn(x, y) for x, y in zip(sizes[1:], sizes[:-1])]
        #ARRAY Per ogni layer L: LxL-1 (quando L=1 rappresentiamo la size dell'input)
        #self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        # i biases e weights sono salvati come liste di matrici per cui - ad esempio - weights[1] sara' una matrice
        # contenente i pesi che connettono il secondo e terzo strato di neuroni

    def forwardPropagation(self, activation, activation_function):
        #for biases, weights in zip(self.biases, self.weights):
            #activation = activation_function(np.dot(weights, activation) + biases)
        # np.dot e' una funzione che effettua la moltiplicazione matriciale

        for weights, biases in zip(self.weights, self.biases):
            #weights: LxL-1 , activation: L-1x1 , w * a : Lx1 , biases.T : Lx1
            activation = activation_function(np.dot(weights, activation) + biases.transpose())
            #activation: Lx1
        return activation

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

    #def backPropagation(self):



def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

#print('---INSERISCI IL NUMERO DI STRATI DELLA RETE---')
#print('---INSERISCI LA FUNZIONE DI OUTPUT---')
#val=input()
val=Network([5, 3, 4, 3])
print('----VAL---')
print(val)
act = np.random.randn(1,5)
print('----INPUT---')
print(act)
print('----APPLICO FORWARD PROPAGATION---')
act=val.forwardPropagation(act.transpose(),sigmoid)
print('----OUTPUT---')
print(act)




