import numpy as np
import warnings


def derivative_least_square(output, target):
    return output - target


def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))


def identity(z):
    return z


def least_square(output, target):
    err = [np.square(o - t) for o, t in zip(output, target)]
    return sum(err)*0.5


def derivative_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

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

def dictionary():
    funzioni = dict()
    funzioni["sigmoid"] = [sigmoid, derivative_sigmoid]
    funzioni["identity"] = [identity, derivative_identity]
    funzioni["cross_entropy"] = [cross_entropy, derivative_cross_entropy]
    funzioni["least_square"] = [least_square, derivative_least_square]
    return funzioni



