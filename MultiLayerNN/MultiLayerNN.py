import numpy as np


class MultiLayerNN():
    """
    Class that implements a multi-layer neural network
    On-going...
    Acknowledgement: based on code from the Neural Networks and 
    Deep Learning Course from Coursera.
    """

    def __init__(self, nn_dimensions):
        # self.nn_dimensions: list containing the dimensions of each layer of the network

        self.nn_dimensions = nn_dimensions
        self.parameters = {}
        self.grads = {}
        self.costs = []

    @staticmethod
    def sigmoid(z):
        """
        Computes the sigmoid of z, and z as a cache value
        """
        return 1/(1+np.exp(-z)),z

    @staticmethod
    def relu(z):
        """
        Computes the relu value of z, and z as a cache value
        """
        return z*(z>0), z

    def train(self, X, y, learning_rate = 0.0075, n_iterations = 3000, print_cost=False):
        
        self.initialize()

        for i in range(0, n_iterations):

            AL, caches = self.forwardPropagation(X)

            cost = compute_cost(AL, y)

            self.grads = self.backwardPropagation(AL, y, caches)

            self.update_parameters(learning_rate)
                
            # Print the cost every 100 training example
            if print_cost and i % 100 == 0:
                print ("Cost after iteration %i: %f" %(i, cost))
                self.costs.append(cost)


    def initialize(self):
        np.random.seed(1)
        L = len(self.nn_dimensions)

        for l in range(1, L):
            self.parameters['W' + str(l)] = np.random.randn(
                self.nn_dimensions[l], self.nn_dimensions[l-1]
                )*0.01
            self.parameters['b' + str(l)] = np.zeros((self.nn_dimensions[l], 1))


    def forwardPropagation(self, X):

        caches = []
        A = X
        L = len(self.nn_dimensions)

        for l in range(1, L):
            A_prev = A 
            A, cache = self.activation_forward(A_prev,
                parameters['W' + str(l)],
                parameters['b' + str(l)],
                "relu"
                )
            caches.append(cache)

        # AL: activation of the final layer
        AL, cache = self.activation_forward(A,
            parameters['W' + str(l)],
            parameters['b' + str(l)],
            "sigmoid"
            )

        # One component of the list caches for each layer of the NN
        caches.append(cache)

        return AL, caches


    def activation_forward(self, A_prev, W, b, activation):

        Z, linear_cache = self.linear_forward(A_prev, W, b)

        if activation == "sigmoid":
            A, activation_cache = self.sigmoid(Z)
        elif activation == "relu":
            A, activation_cache = self.relu(Z)

        cache = (linear_cache, activation_cache)

        return A, cache

    @staticmethod
    def linear_forward(A, W, b):

        Z = np.dot(W,A)+b
        cache = (A, W, b)
        return Z, cache

    @staticmethod
    def compute_cost(AL, y):
        """
        Computes the cross entropy cost function
        """
        m = y.shape[1]
        cost = -1/m*np.sum(np.multiply(np.log(AL),y)+np.multiply(np.log(1-AL),(1-y)))
        cost = np.squeeze(cost)
        return cost

    def backwardPropagation(self, AL, y, caches):

        L = len(self.nn_dimensions)
        m = AL.shape[1]
        y = y.reshape(AL.shape)

        dAL = - (np.divide(y, AL) - np.divide(1 - y, 1 - AL))

        current_cache = caches[L-1]

        self.grads["dA" + str(L-1)], self.grads["dW" + str(L)], self.grads["db" + str(L)] = self.activation_backward(
            dAL,
            current_cache,
            "sigmoid"
            )

        # Loop from l=L-2 to l=0
        for l in reversed(range(L-1)):

            current_cache = caches[l]

            dA_prev_temp, dW_temp, db_temp = self.activation_backward(
                self.grads["dA" + str(l + 1)],
                current_cache,
                "relu"
                )
            self.grads["dA" + str(l)] = dA_prev_temp
            self.grads["dW" + str(l + 1)] = dW_temp
            self.grads["db" + str(l + 1)] = db_temp


    def activation_backward(self, dA, cache, activation):

        linear_cache, activation_cache = cache
    
        if activation == "relu":
            dZ = self.relu_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
            
        elif activation == "sigmoid":
            dZ = self.sigmoid_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
        
        return dA_prev, dW, db

    @staticmethod
    def relu_backward(dA, activation_cache):
        dZ = dA * (activation_cache>0)

    def sigmoid_backward(self, dA, activation_cache):
        dZ = dA * (self.sigmoid(activation_cache)*(1-self.sigmoid(activation_cache)))
        return dZ

    def linear_backward(dZ, linear_cache):
        A_prev, W, b = cache
        m = A_prev.shape[1]

        dW = 1/m*np.dot(dZ,A_prev.T)
        db = 1/m*np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T,dZ)

        return dA_prev, dW, db

    def update_parameters(self, learning_rate):

        L = len(self.nn_dimensions)

        for l in range(L):
            self.parameters["W" + str(l+1)] -= learning_rate*self.grads["dW" + str(l+1)]
            self.parameters["b" + str(l+1)] -= learning_rate*self.grads["db" + str(l+1)]



    
