import numpy as np


class OneHiddenLayerNN():

    def __init__(self, n_hidden_neurons, n_features=2, n_output_neurons=1, n_samples=1):
        self.n_features = n_features
        self.n_hidden_neurons = n_hidden_neurons
        self.n_output_neurons = n_output_neurons
        self.n_samples = n_samples
        self.params = {}
        self.cache = {}
        self.grads = {}

    def initialize(self):
        np.random.seed(1)
        self.params['W1'] = np.random.randn(self.n_hidden_neurons, self.n_features)*0.01
        self.params['b1'] = np.zeros((self.n_hidden_neurons, 1))
        self.params['W2'] = np.random.randn(self.n_output_neurons, self.n_hidden_neurons)*0.01
        self.params['b2'] = np.zeros((n_output_neurons, 1))

    @staticmethod
    def sigmoid(z):
        """
        Computes the sigmoid of z
        """
        return 1/(1+np.exp(-z))

    def forward_propagation(self, X):
        self.cache['Z1'] = np.dot(self.params['W1'],X)+self.params['b1']
        self.cache['A1'] = np.tanh(self.cache['Z1'])
        self.cache['Z2'] = np.dot(self.params['W2'],self.cache['A1'])+self.params['b2']
        self.cache['A2'] = sigmoid(self.cache['Z2'])

    def compute_cost(self, y):
        m = y.shape[1] # number of example
        # Compute the cross-entropy cost
        logprobs = np.multiply(
            np.log(self.cache['A2']),y)+np.multiply(np.log(1-self.cache['A2']),
            (1-y)
            )
        cost = -1/m*np.sum(logprobs)
        cost = float(np.squeeze(cost))  # makes sure cost is the dimension we expect. 
        return cost

    def backward_propagation(self, X, Y):
        m = X.shape[1]
        dZ2 = A2-Y
        self.grads['dW2'] = 1/m*np.dot(dZ2,self.cache['A1'].T)
        self.grads['db2'] = 1/m*np.sum(dZ2, axis=1, keepdims=True)
        dZ1 = np.multiply(np.dot(self.params['W2'].T,dZ2),(1 - np.power(self.cache['A1'], 2)))
        self.grads['dW1'] = 1/m*np.dot(dZ1,X.T)
        self.grads['db1'] = 1/m*np.sum(dZ1, axis=1, keepdims=True)

    def update_parameters(self, learning_rate):
        self.params['W1'] = learning_rate*self.grads['dW1']
        self.params['b1'] = learning_rate*self.grads['db1']
        self.params['W2'] = learning_rate*self.grads['dW2']
        self.params['b2'] = learning_rate*self.grads['db2']

    def train(X, y, num_iterations = 10000, learning_rate = 1.2, print_cost=False):
        self.n_features = X.shape[0]
        self.n_samples = X.shape[1]
        self.n_output_neurons = y.shape[0]
        
        # Initialize parameters
        self.params = self.initialize()

        for i in range(0, num_iterations):
         
            # Forward propagation. 
            self.forward_propagation(X)
            
            # Compute Cost function.
            cost = compute_cost(y)
     
            # Backpropagation. 
            self.backward_propagation(X, y)
     
            # Gradient descent parameter update. 
            self.update_parameters(learning_rate)
            
            # Print the cost every 1000 iterations
            if print_cost and i % 1000 == 0:
                print ("Cost after iteration %i: %f" %(i, cost))
    

    

