import numpy as np


def derivative_least_square(output, target):
    return np.array([o - t for o, t in zip(output, target)])


def sigmoid(x):
    return np.array([1.0 / (1.0 + np.exp(-z)) for z in x])


def identity(z):
    return z


def least_square(output, target):
    err = [np.square(o - t) for o, t in zip(output, target)]
    return sum(err)*0.5


def derivative_sigmoid(x):
    return np.array([z * (1 - z) for z in x])


def softmax(output):
    return np.exp(output) / np.sum(np.exp(output))


def derivative_cross_entropy(output, target):
    output = softmax(output)
    return (output - target) / (output * (1 - output))


def derivative_identity(z):
    return 1


def cross_entropy(output, target):
    output = softmax(output)
    err = [(t * np.log(o) + (1 - t) * np.log(1 - o)) for o, t in zip(output, target)]
    return - sum(err)




