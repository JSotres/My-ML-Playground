import numpy as np
import matplotlib.pyplot as plt


class MultiLayerNN():
    """
    Class that implements a multi-layer neural network
    On-going...
    Acknowledgement: based on code from the Neural Networks and 
    Deep Learning Course from Coursera.
    """

    def __init__(self, dimensions):
        # self.nn_dimensions: list containing the dimensions of each layer of the network

        self.n_dimensions = len(dimensions)
        self.dimensions = dimensions
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

    def train(self, X, y, print_cost=False, learning_rate = 0.0075, n_iterations = 30000):
        
        self.initialize()

        for i in range(0, n_iterations):

            AL, caches = self.forwardPropagation(X)

            cost = self.compute_cost(AL, y)

            self.backwardPropagation(AL, y, caches)

            self.update_parameters(learning_rate)
                
            # Print the cost every 1000 training example
            if print_cost and i % 1000 == 0:
                print ("Cost after iteration %i: %f" %(i, cost))
                self.costs.append(cost)


    def initialize(self):
        np.random.seed(1)
        L = len(self.dimensions)

        for l in range(1, L):
            self.parameters['W' + str(l)] = np.random.randn(
                self.dimensions[l], self.dimensions[l-1]
                )*0.01
            self.parameters['b' + str(l)] = np.zeros((self.dimensions[l], 1))


    def forwardPropagation(self, X):

        caches = []
        A = X
        L = len(self.parameters) // 2

        for l in range(1, L):
            A_prev = A 
            A, cache = self.activation_forward(A_prev,
                self.parameters['W' + str(l)],
                self.parameters['b' + str(l)],
                "relu"
                )
            caches.append(cache)

        # AL: activation of the final layer
        AL, cache = self.activation_forward(A,
            self.parameters['W' + str(L)],
            self.parameters['b' + str(L)],
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

        L = len(self.parameters) // 2
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
        return dZ

    def sigmoid_backward(self, dA, activation_cache):
        s, _s_ = self.sigmoid(activation_cache)
        dZ = dA * (s*(1-s))
        return dZ

    @staticmethod
    def linear_backward(dZ, linear_cache):
        A_prev, W, b = linear_cache
        m = A_prev.shape[1]
        
        dW = 1/m*np.dot(dZ,A_prev.T)
        db = 1/m*np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T,dZ)

        return dA_prev, dW, db

    def update_parameters(self, learning_rate):

        L = len(self.parameters) // 2

        for l in range(L):
            self.parameters["W" + str(l+1)] -= learning_rate*self.grads["dW" + str(l+1)]
            self.parameters["b" + str(l+1)] -= learning_rate*self.grads["db" + str(l+1)]

    def predict(self, X):
        AL, _ = self.forwardPropagation(X)
        y_predicted = np.zeros((1, AL.shape[1]))
        for i in range(AL.shape[1]):
            y_predicted[0,i] = AL[0,i] > 0.5

        return y_predicted

    def scoring(self, y_predicted, y_true):
        """
        Returns accuracy
        """
        return((y_predicted == y_true).sum()/y_predicted.shape[1])

    def plotCosts(self):
        # Plot learning curve (with costs)
        costs = np.squeeze(self.costs)
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('iterations/1000')
        plt.title("Cost Function")
        plt.show()



    
