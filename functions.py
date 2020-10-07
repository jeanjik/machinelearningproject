import numpy as np


def derivative_least_square(output, target):
    return output - target


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def identity(z):
    return z


def least_square(output, target):
    err = [np.square(o - t) for o, t in zip(output, target)]
    return sum(err)*0.5


def derivative_sigmoid(z):
    return z * (1 - z)


def softmax(output):
    return np.exp(output) / np.sum(np.exp(output))


def cross_entropy(output, target):
    err = [(t * np.log(o) + (1 - t) * np.log(1 - o)) for o, t in zip(output, target)]
    return - sum(err)




