import numpy as np
import matplotlib.pyplot as plt


class LogisticRegressionModel():
    """
    Class that implements a simple logistic regression model.
    Optimization algoritm: gradient descent.
    Does not implement regularization (yet).
    Acknowledgement: based on code from the Neural Networks and 
    Deep Learning Course from Coursera.

    ATTRIBUTES:
    self.dim: model dimensions/number of features
    self.params: a dictionary. The first value, weights,
        corresponds to a numpy array of shape (number of features, 1) 
        and the second value, the bias, are all initialized 
        with 0 values.
    self.gradients: a dictionary. The first value corresponds to a
        numpy array, dw, of shape (number of features, 1) that corresponds to
        the derivative of the cost function with respect to each of
        the weights. The second value corresponds to a scalar, db, that
        corresponds to the derivative of the cost function with respect
        to the bias.
    self.costs: numpy array of the values of the cost function obtained
        every 1000 iterations during training.

    METHODS:
    __init__(self, n_features):
    initialize(n_features):
    sigmoid(z):
    """

    def __init__(self, n_features):
        self.n_features = n_features
        self.params = {}
        self.gradients = {}
        self.costs = []
        
    @staticmethod
    def initialize(n_features):
        """
        This function creates a dictionary, params, where the first value, weights,
        corresponds to a numpy array of shape (self.n_features, 1) and the second 
        value, the bias, are all initialized with 0 values.        
        """
        w = np.zeros((n_features,1))
        b = 0
        params = {"w": w, "b": b}
        return params

    @staticmethod
    def sigmoid(z):
        """
        Computes the sigmoid of z

        Arguments:
        z -- A scalar or numpy array of any size.

        Returns:
        s -- sigmoid(z)
        """

        s = 1/(1+np.exp(-z))
        return s

    def propagate(self, X, y):
        """
        Calculates the cost function and its gradient for 
        the forward and backward propagation

        Arguments:
        X -- data of size (number of features, number of examples)
        Y -- true "label" array (values 0 or 1) of shape (1, number of examples)

        Return:
        cost -- negative log-likelihood cost for logistic regression

        Updates:
        self.gradients
        """

        m = X.shape[1]
        

        # FORWARD PROPAGATION
        # compute activation
        A = self.sigmoid(np.dot(self.params["w"].T,X)+self.params["b"])
        # compute cost  
        cost = -1/m*np.sum(y*np.log(A)+(1-y)*np.log(1-A))   

        # BACKWARD PROPAGATION
        self.gradients["dw"] = np.dot(X,(A-y).T)/m
        self.gradients["db"] = np.sum(A-y)/m

        cost = np.squeeze(cost)

        return cost


    def train(self, X, Y, num_iterations, learning_rate, print_cost = False):
        """
        This function trains the logistic regression model

        Arguments:
        X -- data of shape (number of features, number of samples)
        Y -- true "label" vector (values 0 or 1), of shape (1, number of samples)
        num_iterations -- number of iterations of the optimization loop
        learning_rate -- learning rate of the gradient descent update rule
        print_cost -- True to print the loss every 1000 steps

        """

        # Initializes  the costs and params attributes    
        self.costs = []
        self.params = self.initialize(self.n_features)
    
        for i in range(num_iterations):
            # Perform forward and backward propagation. Also calculates, and returns,
            # the Cost function
            cost = self.propagate(X, Y)

            # updates weights and bias
            self.params["w"] -= learning_rate*self.gradients["dw"]
            self.params["b"] -= learning_rate*self.gradients["db"]

            # Record the costs every 1000 iterations
            if i % 1000 == 0:
                self.costs.append(cost)

            # Print on the screen the cost every 1000 training iterations if the 
            # input variable print_cost is set to True
            if print_cost and i % 1000 == 0:
                print ("Cost after iteration %i: %f" %(i, cost))


    def predict(self, X):
        '''
        Predict whether the label is 0 or 1 using learned 
        logistic regression parameters

        Arguments:
        X -- data of size (number of features, number of examples)

        Returns:
        y_prediction -- a numpy array (vector) containing all 
        predictions (0/1) for the examples in X
        '''

        m = X.shape[1]
        y_prediction = np.zeros((1,m))
        w = self.params['w'].reshape(X.shape[0], 1)
        b = self.params['b']

        # Compute vector "A" predicting the probabilities of a X being a 1
        A = self.sigmoid(np.dot(w.T,X)+b)

        for i in range(A.shape[1]):
            y_prediction[0,i] = A[0,i] > 0.5

        return y_prediction

    def plotCosts(self):
        # Plot learning curve (with costs)
        costs = np.squeeze(self.costs)
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('iterations/1000')
        plt.title("Cost Function")
        plt.show()

    def scoring(self, y_predicted, y_true):
        """
        Returns accuracy
        """
        return((y_predicted == y_true).sum()/y_predicted.shape[1])




