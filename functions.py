import numpy as np


def derivative_least_square(output, target):
    return output - target


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def identity(z):
    return z


def derivative_sigmoid(z):
    return z * (1 - z)