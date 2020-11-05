import random
import numpy as np
import functions as fn
import loader as ld
np.seterr(divide='ignore', invalid='ignore')

class Network(object):
    #inizializzazione della rete
    def __init__(self, sizes, activation, error):
        #passa in input il num di neuroni di ogni layer: array[input, L1, L2, ...]
        self.num_layers = len(sizes) #numero di strati della rete L
        self.sizes = sizes #array con numero di nodi per ciascun layer
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]  #array di matrici colonna Lx1
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])] #array di matrici
        self.activation_function = activation #funzione di attivazione per ciascun layer
        self.error_function = error #funzione di errore
        self.dictionary_function = fn.dictionary() #creazione del dizionario di funzioni

    def forward_propagation(self, a):
        for b, w, i in zip(self.biases, self.weights, range(0,self.num_layers)):
            #si estra la funzione di attivazione dell'i-esimo layer
            f = self.dictionary_function[self.activation_function[i]][0]
            #z(i) = w(i,i-1) * a(i-1) + b(i)
            #a(i) = f(z)
            a = f(np.dot(w, a) + b)
        #ritorna l'output dell'ultimo strato
        return a

    #calcola le derivate della funzione di errore rispetto ai pesi
    def back_propagation(self, input, target):
        #la derivata dell'errore rispetto al bias e' il delta
        delta_err_b = [np.zeros(b.shape) for b in self.biases]  #matrice dei delta per il bias
        #la derivata dell'errore rispetto ai pesi e' delta * z attivato
        derivative_err_w = [np.zeros(w.shape) for w in self.weights] #matrice delle derivate per i pesi
        #forward propagation
        activation = input
        activations = [input]
        zs = []
        for b, w, i in zip(self.biases, self.weights, range(0, self.num_layers)):
            #z(i) = w(i,i-1) * a(i-1) + b(i)
            z = np.dot(w, activation) + b
            #salva z non attivato
            zs.append(z)
            #si estra la funzione di attivazione dell'i-esimo layer
            f = self.dictionary_function[self.activation_function[i]][0]
            #a(i) = f(z)
            activation = f(z)
            #salva z attivato
            activations.append(activation)

        #passaggio backward per i nodi di output
        if (self.error_function == "cross_entropy" and self.activation_function[-1] == "softmax"):
            #si estrae funzione per la derivata di softmax con cross-entropy
            derivative_cross_entropy_softmax=self.dictionary_function[self.activation_function[-1]][1]
            #calcolo il delta: output - target
            delta_error = derivative_cross_entropy_softmax(activations[-1], target)
            delta_err_b[-1] = delta_error
        else:
            #caso generico
            #si estrae la funzione di errore
            derivative_error_function = self.dictionary_function[self.error_function][1]
            #calcolo la derivata della funzione di errore rispetto all'output (z attivato dell'ultimo strato)
            delta_error = derivative_error_function(activations[-1], target)
            #si estrae la funzione di attivazione dell'ultimo strato
            derivative_activation_function = self.dictionary_function[self.activation_function[-1]][1]
            #calcolo il delta: f'(z non attivato) * derivata della funzione di errore rispetto all'output
            act = derivative_activation_function(zs[-1])
            delta_err_b[-1] = delta_error * act
        #calcolo la derivata: delta * z attivato
        derivative_err_w[-1]=np.dot(delta_error,activations[-2].transpose())

        #per i nodi interni, formula ricorrente a partire dal penultimo strato (-l)
        for l in range(2, self.num_layers):
            z = zs[-l] #z non attivato del - l-esimo strato
            #si estrae la funzione di attivazione del -l strato
            derivative_activation_function = self.dictionary_function[self.activation_function[-l]][1]
            #calcolo la derivata della funzione di errore su z non attivato
            sp = derivative_activation_function(z)

            #calcolo il delta: f'(z non att) * (delta precedente .dot pesi dello strato successivo)
            delta_error = np.dot(self.weights[-l + 1].transpose(), delta_error) * sp
            delta_err_b[-l] = delta_error

            #calcolo la derivata: delta * z attivato dello strato precedente
            derivative_err_w[-l] = np.dot(delta_error, activations[-l - 1].transpose())
        return delta_err_b, derivative_err_w

    def batch_gradient_descent(self, training_data, epochs, eta, momentum, validation_data, k=None):
        #per mantenere traccia dei valori delle epoche precedenti
        pre_w = [np.zeros(w.shape) for w in self.weights]
        pre_b = [np.zeros(b.shape) for b in self.biases]
        err_validation = []
        err_training = []
        #estraggo la funzione di errore
        error_function = self.dictionary_function[self.error_function][0]

        #training_data e' una lista di tuple (x, y) dove x e' l'input e y e' la corrispondente label
        for j in range(epochs):
            random.shuffle(training_data)
            #per le derivate totali
            sum_w = [np.zeros(w.shape) for w in self.weights]
            sum_b = [np.zeros(b.shape) for b in self.biases]
            for input, label in training_data:
                #per ogni input del training calcolo le derivate parziali
                der_b, der_w = self.back_propagation(input, label)

                #aggiorno il valore delle derivate
                sum_b = [sb + derb for sb, derb in zip(sum_b, der_b)]
                sum_w = [sw + derw for sw, derw in zip(sum_w, der_w)]
            #utilizzo un aggiornamento BATCH
            #w = w - (eta * derivata w + momento * derivata dell'epoca precedente)
            self.weights = [w - (eta * wl + momentum * pwl) for w, wl, pwl in zip(self.weights, sum_w, pre_w)]
            self.biases = [b - (eta * bl + momentum * pbl) for b, bl, pbl in zip(self.biases, sum_b, pre_b)]

            #tengo traccia dei valori dell'epoche precedenti
            pre_b = pre_b + sum_b
            pre_w = pre_w + sum_w

            #calcolo l'errore sul validation e training set
            err_validation.append(self.calculate_error(validation_data, error_function))
            err_training.append(self.calculate_error(training_data, error_function))
            print("ERRORE VALIDATION EPOCA " + str(j+1) + str(err_validation[j]) + " SU " + str(len(validation_data)))
            print("ERRORE TRAINING EPOCA " + str(j+1) + str(err_training[j]) + " SU " + str(len(training_data)))
            print("FINE EPOCA" + str(j+1))

            #valuto il criterio di early stopping
            if k is not None:
                stop = self.check_early_stopping(err_validation, j, k)
                if stop is True:
                    return err_validation
        return err_training, err_validation

    def compute_tp_tn_fp_fn(self, test_data):
        target = [np.argmax(y) for (x, y) in test_data]
        prediction = [np.argmax(self.forward_propagation(x)) for (x, y) in test_data]
        values = []
        #calcola i valori TP,TN,FN e FP per ogni classe
        for i in range(0,self.sizes[-1]):
            true_positives = sum((t == i) and (p == i) for t, p in zip(target, prediction))  # veri positivi
            true_negatives = sum((t != i) and (p != i) for t, p in zip(target, prediction))  # veri negativi
            false_negatives = sum((t == i) and (p != i) for t, p in zip(target, prediction))  # falsi negativi
            false_positives = sum((t != i) and (p == i) for t, p in zip(target, prediction))  # falsi positivi
            values.append([true_positives, true_negatives, false_negatives, false_positives])
        return values

    #per classi non equamente distribuite (sbilanciate)
    def microaveraging(self, values):
        #calcola un solo valore di TP,TN,FN e FP
        #per calcolare un solo valore di Accuratezza, Precision, Recall e F1
        sum_tp = 0
        sum_tn = 0
        sum_fn = 0
        sum_fp = 0
        for val in values:
            sum_tp = sum_tp + val[0]
            sum_tn = sum_tn + val[1]
            sum_fn = sum_fn + val[2]
            sum_fp = sum_fp + val[3]
        accuracy = (sum_tp + sum_tn) / float((sum_tp + sum_tn + sum_fp + sum_fn))
        precision = sum_tp / float(sum_tp + sum_fp)
        recall = sum_tp / float(sum_tp + sum_fn)
        f1 = (2 * precision * recall) / float(precision + recall)
        return accuracy, precision, recall, f1

    def macroaveraging(self, values):
        #calcola i valori di Accuratezza, Precision, Recall e F1 per ogni classe
        #per poi ritornare una media dei valori
        accuracy = []
        precision = []
        recall = []
        f1 = []
        for val in values:
            accuracy.append((val[0] + val[1]) / float((val[0] + val[1] + val[3] + val[2])))
            p = val[0] / float(val[0] + val[3])
            precision.append(p)
            r = val[0] / float(val[0] + val[2])
            recall.append(r)
            f1.append((2 * p * r) / float(p + r))
        return np.mean(accuracy), np.mean(precision), np.mean(accuracy), np.mean(f1)

    #criterio di early stopping
    def check_early_stopping(self, err, epoch, k):
        if epoch - k > 0:
            if err[epoch] > err[epoch - k]:
                return True
        return False

    #calcola dell'errore
    def calculate_error(self, test_data, error_function):
        err = 0
        for inp, target in test_data:
            #propaga in avanti
            out = self.forward_propagation(inp)
            #accumula l'errore calcolato mediante la funzione di errore su output e target
            err = err + error_function(out, target)
        #il risultato viene normalizzato
        return err / len(test_data)

    # conta il numero di veri positivi di tutte le classi
    def evaluate(self, test_data):
        test_results = [(np.argmax(self.forward_propagation(x)), np.argmax(y))
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)


if __name__ == "__main__":
    error = "cross_entropy"
    activation = ["sigmoid", "softmax"]
    val = Network([784, 30, 10], activation, error)
    dim = val.sizes[1]
    tr_d = ld.load_data(3000)
    training_set = tr_d[:2000]
    validation_set = tr_d[2000:2500]
    test_set = tr_d[2500:3000]
    epochs = 150; momentum = 0.86; eta = 0.0045; k = 30
    err_training, err_validation = val.batch_gradient_descent(training_set, epochs, eta, momentum, validation_set)
    err_test = val.calculate_error(test_set, val.dictionary_function[val.error_function][0])
    print("CALCOLO ERRORE SU TEST SET" + str(err_test))
    guessed_right = val.evaluate(test_set)
    print("NE HO INDOVINATE " + str(guessed_right))
    values = val.compute_tp_tn_fp_fn(test_set)
    accuracy, precision, recall, f1 = val.macroaveraging(values)
    error_trend = np.asarray([(j+1, tr, val) for j, tr, val in zip(range(epochs), err_training, err_validation)])
    print "MACROAVERAGING"
    print("ACCURATEZZA " + str(accuracy))
    print("PRECISONE " + str(precision))
    print("RECALL " + str(recall))
    print("F1 MEASURE " + str(f1))
    results = np.asarray([[dim, epochs, momentum, eta, err_training[-1], err_validation[-1], err_test,
                           guessed_right, accuracy, precision, recall, f1]])
    #per l'analisi dell'errore:
    with open("./results/res.csv", "ab") as f:
        np.savetxt(f, results, delimiter=",", fmt="%1.6f")
    error_trend_file_path = "./results/error_trend_test21.csv"
    with open(error_trend_file_path, "w+") as f:
        np.savetxt(f, error_trend, delimiter=",", fmt="%1.6f")

