---
categories: kaggle machine-learning
date: '2021-10-08'
layout: post
slug: mathematics-of-deep-neural-nets
title: Mathematics Of Deep Neural Nets
---

# Mathematics of Deep Neural Nets

Neural Nets are defined by a **learning function** $F(x,v)$ where the **weights** are $x$ and the **training data** are $v$.

The most important aspects of creating $F$: 

1. Composition $ F = F_3(F_2(F_1(x,v,)))$
2. Chain Rule for $x$-derivatives of $F$
3. Stochastic Gradient Descent (SGB) to find the best weights $x$ 
4. Backpropagation to execute th chain rule 
5. Introudce nonlinearity with the Rectified Linear Unit function $ \text{ReLU}(y) = \text{max}(y, 0) = \text{ramp function}$ 

The layers of Neural Net are $F_1, F_2, F_3$, and the weights $x$ that connected the layers $v$ are optimized in creating $F$. 

# $F_k$ is a Piecewise Linear Function of $v_{k-1}$ 

The input to $F_k$ is a vector $v_{k-1}$ of length $N_{k-1}$, where $N_k$ is the number of neurons in the layer. The output is a vector $v_k$ of length $N_k$, ready for input to $F_{k+1}$. This function $F_k$ has two parts: 

<!-- , first linear (multiplying by weights) and then nonlinear (element wise multiplication by activation fucntion).  -->

1. The **linear** part of $F_k$ yields $A_k v_k + b_k$ (b_k is a **bias vector** and makes this "affine")
2. The **non-linear** part is our activation function (ReLU) is applied to *each component* (element wise) to $A_k v_k + b_k$

The training data for each sample is a in a **feature vector** $v_0$. THe matrix $A_k$ has shape $N_k$ (rows) by $N_{k-1}$ (columns). The column vector $b_k$ has $N_k$ componentes. **These $A_k$ and $b_k$ are weights constructed by the optimization algorithm.** Frequently we use Stochastic Gradient Descent(SGD) to compute the **optimal weights** where $ x = (A_1, b_1, ..., A_L, b_L)$. SGB is the central computation of deep learning. It relies on backpropagation to find the $x$ derivatives of $F$, to solve $\nabla F = 0$. 

the activation function $ \text{ReLU}(y) = \text{max}(y, 0)$ gives flexibility and adaptability. Linear steps alone were not enough and unsuccessful. 

ReLU is applied to every "neuron" in every internal layer. There are $N_k$ neurons in layer k, contain the $N_k$ outputs from $A_k v_k + b_k$. Notice that ReLU itself is continuous and piecewise linear. When we choose ReLU the composite function $F = F_L(F_2(F_1(x, v)))$ has an important property: 

**The learning function $F$ is continuous and piecewise linear in $v$** 

# Example of One Internal Layer($L=2$) 

Suppose we have measured $m = 3$ features of one sample point in the training set. Those features are the 3 components of the inputer vector $v = v_0$. Then the first function $F_1$ in the chain multiplies $v_0$ by a matrix $A_1$ and adds an offset(bias) vector $b_1$. If $A_1$ is 4 (rows) by 3 (columns) and the vector $b_1$ is 4 (rows) by 1 (column) then we have 4 components of $A_0 v_0 + b_0$. 

That step found 4 cominbations of the 3 original features in $v = v_0$. The 12 weights in hte matrix $A_1$ were optimized over many feature vectors $v_0$ in the training set, to choose a 4 x 3 matrix ( a 4 by 1 bias vector) that would find 4 insightful combinations. 

The final sep to reach $v_1$ is to apply the nonlinear "activation function" to each of the 4 components of $A_1 v_0 + b_1$. 


```python
%matplotlib inline
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np
import math

def dReLU(x):
    return 1. * (x > 0)

x = np.linspace(-4, 4, 1000)
y = np.maximum(0, x)
derivative_relu = dReLU(x)

plt.figure(figsize=(10, 5))
plt.plot(x, y)
plt.plot(x, derivative_relu)
plt.legend(['ReLU', 'derivative ReLU'])
plt.show()
```

Historically it was thought a sudden change of slope would be dangerous and possibly unstable. But large scale numerical experiments indicated otherwise! A better result was achieved by the **ramp function (ReLU)**


So what we want to do is substitute $A_1 v_0 + b_1$ into ReLU to find $v_1$. We end up with: 
$$ (v_1)_k = \text{max}((A_1 v_0 + b_1)_k, 0) $$

Now we have the components of $v_1$ at the four ($N_k = 4$) in layer 1. The input layer held the three components of this particular sample of training idata. We may have thousands or millions of samples. The optimizatino algorithm found $A_1$ and $b_1$ and possibly by SGD using backpropagation to compute gradients of the overall *loss*. 

Suppose our neural net is shallow instead of deep. It has this first layer of 4 neurons. Then the final step will multple the 4-component vector $v_1$ by a 1 by 4 matrix $A_2$ (a row vector). It can add a single number $b_2$ to reach th value $v_2 = A_2 v_1 + b_2$. The nonlinear function ReLU is not applied to the output. 

**Overall we compute $v2 = F(x, v_0)$ for each feature vector $v_0$ in the training set. the steps are $v_2 = A_2 v_1 + b_2 = A_2(\text{ReLU}(A_1 v_0 + b_1)) + b_2 = F(x, v_0)$**

The goal in optimizing $ x = A_1, b_1, A_2, b_2$ is that the output values $v_l = v_2$ at the last layer ($l = 2$) should correctly capture the important features of the training data $v_0$. 

![feed_forward_neural_net.png](attachment:6034da16-e73c-4ba9-a8dd-9a9361b9e746.png)

* For a **classfication problem** each sample v_0 of the training data is assigned **1 or -1**. We want the output $v_2$ to have that correct sign. 

* For a **regression problem** we use the numerical value of $v_2$. We do not choose enough weights $A_k$ and $b_k$ to *get every sample correct*. And we don't necessarily want to! That would result in *overfitting*. 

Depending on our choice of loss function $L(x, v_2)$ to minimize, this problem can be like least square (square loss) or entropy minimization (cross-entropy loss)). We are choose $ x = A_k \text{weight matrices} and b_k \text{bias vectors}$ to minimize our loss function $L$. 


Our hope is that **the function $F$ has "learned" the data.** We don't want to choose so many weights in $x$ that every input sample is sure to be correctly classified because that *is not learning*. That is just *overfitting*. 

We are looking for a balance where the function $F$ has learned what is important in recognizing the output e.g. dog vs cat. 

Machine Learning doesn't aim to capture every detail of the numbers 0, 1, 2,...,9. It just aims to capture enough information to decide correctly *which number it is*. 

All of this is taken from Strang "Learning From Data" (2019) 

# Coding It in Python


```python
import numpy as np
```


```python
class FullyConnectedLayer:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(input_size, output_size) / np.sqrt(input_size + output_size)
        self.bias = np.random.randn(1, output_size) / np.sqrt(input_size + output_size)

    def forward(self, input):
        self.input = input
        return np.dot(input, self.weights) + self.bias

    def backward(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        # bias_error = output_error
        
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error
```


```python
class ActivationLayer:
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime
    
    def forward(self, input):
        self.input = input
        return self.activation(input)
    
    def backward(self, output_error, learning_rate):
        return output_error * self.activation_prime(self.input)
```


```python
class FlattenLayer:
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def forward(self, input):
        return np.reshape(input, (1, -1))
    
    def backward(self, output_error, learning_rate):
        return np.reshape(output_error, self.input_shape)
```


```python
class SoftmaxLayer:
    def __init__(self, input_size):
        self.input_size = input_size
    
    def forward(self, input):
        self.input = input
        tmp = np.exp(input)
        self.output = tmp / np.sum(tmp)
        return self.output
    
    def backward(self, output_error, learning_rate):
        input_error = np.zeros(output_error.shape)
        out = np.tile(self.output.T, self.input_size)
        return self.output * np.dot(output_error, np.identity(self.input_size) - out)
```


```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    return np.exp(-x) / (1 + np.exp(-x))**2

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1 - np.tanh(x)**2

def relu(x):
    return np.maximum(x, 0)

def relu_prime(x):
    return np.array(x >= 0).astype('int')
```


```python
def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_pred.size

def sse(y_true, y_pred):
    return 0.5 * np.sum(np.power(y_true - y_pred, 2))

def sse_prime(y_true, y_pred):
    return y_pred - y_true
```

# Testing with XOR


```python
class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    # add layer to network
    def add(self, layer):
        self.layers.append(layer)

    # set loss to use
    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    # predict output for given input
    def predict(self, input_data):
        # sample dimension first
        samples = len(input_data)
        result = []

        # run network over all samples
        for i in range(samples):
            # forward propagation
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward(output)
            result.append(output)

        return result

    # train the network
    def fit(self, x_train, y_train, epochs, learning_rate):
        # sample dimension first
        samples = len(x_train)

        # training loop
        for i in range(epochs):
            err = 0
            for j in range(samples):
                # forward propagation
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward(output)

                # compute loss (for display purpose only)
                err += self.loss(y_train[j], output)

                # backward propagation
                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward(error, learning_rate)

            # calculate average error on all samples
            err /= samples
            print('epoch %d/%d   error=%f' % (i+1, epochs, err))
```


```python
import numpy as np

# from network import Network
# from fc_layer import FCLayer
# from activation_layer import ActivationLayer
# from activations import tanh, tanh_prime
# from losses import mse, mse_prime

# training data
x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

# network
net = Network()
net.add(FullyConnectedLayer(2, 3))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FullyConnectedLayer(3, 1))
net.add(ActivationLayer(tanh, tanh_prime))

# train
net.use(mse, mse_prime)
net.fit(x_train, y_train, epochs=1000, learning_rate=0.1)

# test
out = net.predict(x_train)
print(out)
```


```python
from tensorflow.keras.datasets import mnist
from tensorflow.keras import utils

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32')
x_train /= 255
y_train = utils.to_categorical(y_train)
x_train = x_train[0:1000]
y_train = y_train[0:1000]

x_test = x_test.astype('float32')
x_test /= 255
y_test = utils.to_categorical(y_test)
```


```python
network = [
    FlattenLayer(input_shape=(28, 28)),
    FullyConnectedLayer(28 * 28, 128),
    ActivationLayer(relu, relu_prime),
    FullyConnectedLayer(128, 10),
    SoftmaxLayer(10)
]

epochs = 40
learning_rate = 0.1

# training
for epoch in range(epochs):
    error = 0
    for x, y_true in zip(x_train, y_train):
        # forward
        output = x
        for layer in network:
            output = layer.forward(output)
        
        # error (display purpose only)
        error += mse(y_true, output)

        # backward
        output_error = mse_prime(y_true, output)
        for layer in reversed(network):
            output_error = layer.backward(output_error, learning_rate)
    
    error /= len(x_train)
    print('%d/%d, error=%f' % (epoch + 1, epochs, error))
```


```python
def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output

ratio = sum([np.argmax(y) == np.argmax(predict(network, x)) for x, y in zip(x_test, y_test)]) / len(x_test)
error = sum([mse(y, predict(network, x)) for x, y in zip(x_test, y_test)]) / len(x_test)
print('ratio: %.2f' % ratio)
print('mse: %.4f' % error)
```


```python
import matplotlib.pyplot as plt

samples = 10
for test, true in zip(x_test[:samples], y_test[:samples]):
    image = np.reshape(test, (28, 28))
    plt.imshow(image, cmap='binary')
    plt.show()
    pred = predict(network, test)[0]
    idx = np.argmax(pred)
    idx_true = np.argmax(true)
    print('pred: %s, prob: %.2f, true: %d' % (idx, pred[idx], idx_true))
```
