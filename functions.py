import numpy as np

def derivative_least_square(output, target):
    return output - target

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def identity(z):
    return z

def least_square(output, target):
    err=np.square(output - target)
    return sum(err)*0.5

def derivative_identity(z):
    return 1

def derivative_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

def cross_entropy(output, target):
    return - sum(target * np.log(output))

def derivative_cross_entropy(output, target):
     return (output - target) / (output * (1 - output))

def softmax(output):
    val=np.exp(output)
    return val / np.sum(val)

def derivative_cross_entropy_softmax(output, target):
    return output - target

def dictionary():
    funzioni = dict()
    funzioni["sigmoid"] = [sigmoid, derivative_sigmoid]
    funzioni["identity"] = [identity, derivative_identity]
    funzioni["cross_entropy"] = [cross_entropy, derivative_cross_entropy]
    funzioni["least_square"] = [least_square, derivative_least_square]
    funzioni["softmax"] = [softmax, derivative_cross_entropy_softmax]
    return funzioni



