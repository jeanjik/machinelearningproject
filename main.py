import random
import numpy as np
import functions as fn
import loader as ld
np.seterr(divide='ignore', invalid='ignore')

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
        return a

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
        if (self.error_function == "cross_entropy" and self.activation_function[-1] == "softmax"):
            derivative_cross_entropy_softmax=self.dictionary_function[self.activation_function[-1]][1]
            delta_error = derivative_cross_entropy_softmax(activations[-1], target)
            delta_err_b[-1] = delta_error
        else:
            derivative_error_function = self.dictionary_function[self.error_function][1]
            delta_error = derivative_error_function(activations[-1], target)
            derivative_activation_function = self.dictionary_function[self.activation_function[-1]][1]
            act = derivative_activation_function(zs[-1])
            delta_err_b[-1] = delta_error * act
        derivative_err_w[-1]=np.dot(delta_error,activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            derivative_activation_function = self.dictionary_function[self.activation_function[-l]][1]
            sp = derivative_activation_function(z)
            delta_error = np.dot(self.weights[-l + 1].transpose(), delta_error) * sp
            delta_err_b[-l] = delta_error
            derivative_err_w[-l] = np.dot(delta_error, activations[-l - 1].transpose())
        return delta_err_b, derivative_err_w

    def batch_gradient_descent(self, training_data, epochs, eta, momentum, validation_data, k=None):
        # training_data e' una lista di tuple (x, y) dove x e' l'input e y e' la corrispondente label
        pre_w = [np.zeros(w.shape) for w in self.weights]  # per l'iterazione precedente
        pre_b = [np.zeros(b.shape) for b in self.biases]
        err_validation = []
        err_training = []
        error_function = self.dictionary_function[self.error_function][0]
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
            # print("EPOCA " + str(j+1) + " SONO CORRETTE " + str(self.evaluate(validation_data)) + " SU "
            #      + str(len(validation_data)))
            err_validation.append(self.calculate_error(validation_data, error_function))
            err_training.append(self.calculate_error(training_data, error_function))
            print("ERRORE VALIDATION EPOCA " + str(j+1) + str(err_validation[j]) + " SU " + str(len(validation_data)))
            print("ERRORE TRAINING EPOCA " + str(j+1) + str(err_training[j]) + " SU " + str(len(training_data)))
            print("FINE EPOCA" + str(j+1))
            if k is not None:
                stop = self.check_early_stopping(err_validation, j, k)
                if stop is True:
                    return err_validation
        return err_training, err_validation

    def compute_tp_tn_fp_fn(self, test_data):
        target = [np.argmax(y) for (x, y) in test_data]
        prediction = [np.argmax(self.forward_propagation(x)) for (x, y) in test_data]
        values = []
        for i in range(0,self.sizes[-1]):
            true_positives = sum((t == i) and (p == i) for t, p in zip(target, prediction)) #veri positivi
            true_negatives = sum((t != i) and (p != i) for t, p in zip(target, prediction)) #veri negativi
            false_negatives = sum((t == i) and (p != i) for t, p in zip(target, prediction)) #falsi negativi
            false_positives = sum((t != i) and (p == i) for t, p in zip(target, prediction)) #falsi positivi
            values.append([true_positives, true_negatives, false_negatives, false_positives])
        return values

    def microaveraging(self, values):
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

    def check_early_stopping(self, err, epoch, k):
        if epoch - k > 0:
            if err[epoch] > err[epoch - k]:
                return True
        return False

    def calculate_error(self, test_data, error_function):
        err = 0
        for inp, target in test_data:
            out = self.forward_propagation(inp)
            err = err + error_function(out, target)
        return err / len(test_data)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.forward_propagation(x)), np.argmax(y))
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)


if __name__ == "__main__":
    error = "cross_entropy"
    activation = ["sigmoid", "softmax"]
    val = Network([784, 300, 10], activation, error)
    dim = val.sizes[1]
    tr_d = ld.load_data(3000)
    training_set = tr_d[:2000]
    validation_set = tr_d[2000:2500]
    test_set = tr_d[2500:3000]
    epochs = 150; momentum = 0.86; eta = 0.0026#; k = 30
    err_training, err_validation = val.batch_gradient_descent(training_set, epochs, eta, momentum, validation_set)
    err_test = val.calculate_error(test_set, val.dictionary_function[val.error_function][0])
    print("CALCOLO ERRORE SU TEST SET" + str(err_test))
    guessed_right = val.evaluate(test_set)
    print("NE HO INDOVINATE " + str(guessed_right))
    values = val.compute_tp_tn_fp_fn(test_set)
    # accuracy, precision, recall, f1 = val.microaveraging(values)
    # print "MICROAVERAGING"
    # print("ACCURATEZZA " + str(accuracy))
    # print("PRECISONE " + str(precision))
    # print("RECALL " + str(recall))
    # print("F1 MEASURE " + str(f1))
    accuracy, precision, recall, f1 = val.macroaveraging(values)
    error_trend = np.asarray([(j+1, tr, val) for j, tr, val in zip(range(epochs), err_training, err_validation)])
    print "MACROAVERAGING"
    print("ACCURATEZZA " + str(accuracy))
    print("PRECISONE " + str(precision))
    print("RECALL " + str(recall))
    print("F1 MEASURE " + str(f1))
    results = np.asarray([[dim, epochs, momentum, eta, err_training[-1], err_validation[-1], err_test,
                           guessed_right, accuracy, precision, recall, f1]])
    with open("./results/res.csv", "ab") as f:
        np.savetxt(f, results, delimiter=",", fmt="%1.6f")
    error_trend_file_path = "./results/error_trend_test10.csv"
    with open(error_trend_file_path, "w+") as f:
        np.savetxt(f, error_trend, delimiter=",", fmt="%1.6f")

