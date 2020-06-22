import numpy as np
import matplotlib.pyplot as plt


class LogisticRegressionModel():

    def __init__(self, dim):
        self.dim = dim
        self.params = {}
        self.grads = {}
        self.costs = []
        
    @staticmethod
    def initialize(dim):
        """
        This function creates a dictionary, params, where the first value, wights,
        corresponds to a numpy array of shape (self.dim, 1) and the second value, the bias,
        are all initialized with 0 values.        
        """
        w = np.zeros((dim,1))
        b = 0
        params = {"w": w, "b": b}
        return params

    @staticmethod
    def sigmoid(z):
        """
        Compute the sigmoid of z

        Arguments:
        z -- A scalar or numpy array of any size.

        Return:
        s -- sigmoid(z)
        """

        s = 1/(1+np.exp(-z))
        return s

    def propagate(self, w, b, X, Y):
        """
        Implement the cost function and its gradient for 
        the forward and bakward propagation

        Arguments:
        X -- data of size (number of features, number of examples)
        Y -- true "label" array (values 0 or 1) of shape (1, number of examples)

        Return:
        cost -- negative log-likelihood cost for logistic regression
        dw -- gradient of the loss with respect to w, thus same shape as w
        db -- gradient of the loss with respect to b, thus same shape as b
        """

        m = X.shape[1]
        


        # FORWARD PROPAGATION (FROM X TO COST)
        A = self.sigmoid(np.dot(w.T,X)+b)    # compute activation
        cost = -1/m*np.sum(Y*np.log(A)+(1-Y)*np.log(1-A))   # compute cost

        # BACKWARD PROPAGATION (TO FIND GRAD)
        dw = np.dot(X,(A-Y).T)/m
        db = np.sum(A-Y)/m

        cost = np.squeeze(cost)

        grads = {"dw": dw, "db": db}

        return grads, cost


    def train(self, X, Y, num_iterations, learning_rate, print_cost = False):
        """
        This function optimizes w and b by running a gradient descent algorithm

        Arguments:
        X -- data of shape (num_features, number of examples)
        Y -- true "label" vector (values 0 or 1), of shape (1, number of examples)
        num_iterations -- number of iterations of the optimization loop
        learning_rate -- learning rate of the gradient descent update rule
        print_cost -- True to print the loss every 100 steps

        Returns:
        params -- dictionary containing the weights w and bias b
        grads -- dictionary containing the gradients of the weights 
        and bias with respect to the cost function
        costs -- list of all the costs computed during the optimization, 
        this will be used to plot the learning curve.
        """
    
        self.costs = []
        self.params = self.initialize(self.dim)
    
        for i in range(num_iterations):
            # Cost and gradient calculation (≈ 1-4 lines of code)
            grads, cost = self.propagate(self.params['w'], self.params['b'], X, Y)

            # Retrieve derivatives from grads
            dw = grads["dw"]
            db = grads["db"]

            # update rule (≈ 2 lines of code)
            self.params['w'] -= learning_rate*dw
            self.params['b'] -= learning_rate*db

            # Record the costs
            if i % 100 == 0:
                self.costs.append(cost)

            # Print the cost every 100 training iterations
            if print_cost and i % 1000 == 0:
                print ("Cost after iteration %i: %f" %(i, cost))
    
        #self.params = {"w": w, "b": b}

        self.grads = {"dw": dw, "db": db}


    def predict(self, X):
        '''
        Predict whether the label is 0 or 1 using learned 
        logistic regression parameters (w, b)

        Arguments:
        X -- data of size (num_px * num_px * 3, number of examples)

        Returns:
        Y_prediction -- a numpy array (vector) containing all 
        predictions (0/1) for the examples in X
        '''

        m = X.shape[1]
        Y_prediction = np.zeros((1,m))
        w = self.params['w'].reshape(X.shape[0], 1)
        b = self.params['b']

        # Compute vector "A" predicting the probabilities of a X being a 1
        A = self.sigmoid(np.dot(w.T,X)+b)

        for i in range(A.shape[1]):
            Y_prediction[0,i] = A[0,i] > 0.5

        return Y_prediction

    def plotCosts(self):
        # Plot learning curve (with costs)
        costs = np.squeeze(self.costs)
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('iterations (per hundreds)')
        plt.title("Cost Function")
        plt.show()

    def scoring(self, y_predicted, y_true):
        return((y_predicted == y_true).sum()/y_predicted.shape[1])




