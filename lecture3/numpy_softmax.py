"""MNIST softmax completely in numpy."""
import numpy as np
from tinyflow.datasets import get_mnist

def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    x = np.exp(x)
    x = x / np.sum(x, axis=1, keepdims=True)
    return x

def evaluate(x, y_, W):
    y = softmax(np.dot(x, W))
    return np.mean(np.argmax(y, 1) == np.argmax(y_, 1))

# get the mnist dataset
mnist = get_mnist(flatten=True, onehot=True)

learning_rate = 0.5 / 100
W = np.zeros((784, 10))

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    # forward
    y = softmax(np.dot(batch_xs, W))
    # backward
    y_grad = y - batch_ys
    W_grad = np.dot(batch_xs.T, y_grad)
    # update
    W = W - learning_rate * W_grad

# evaluate
print(evaluate(mnist.test.images, mnist.test.labels, W))
