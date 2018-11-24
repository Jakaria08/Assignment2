from __future__ import division  # floating point division
import numpy as np
import math
import time
import utilities as utils

class Regressor:
    """
    Generic regression interface; returns random regressor
    Random regressor randomly selects w from a Gaussian distribution
    """

    def __init__( self, parameters={} ):
        """ Params can contain any useful parameters for the algorithm """
        self.params = {}
        self.reset(parameters)

    def reset(self, parameters):
        """ Reset learner """
        self.weights = None
        self.resetparams(parameters)

    def resetparams(self, parameters):
        """ Can pass parameters to reset with new parameters """
        self.weights = None
        try:
            utils.update_dictionary_items(self.params,parameters)
        except AttributeError:
            # Variable self.params does not exist, so not updated
            # Create an empty set of params for future reference
            self.params = {}

    def getparams(self):
        return self.params

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        self.weights = np.random.rand(Xtrain.shape[1])

    def predict(self, Xtest):
        """ Most regressors return a dot product for the prediction """
        ytest = np.dot(Xtest, self.weights)
        return ytest

class RangePredictor(Regressor):
    """
    Random predictor randomly selects value between max and min in training set.
    """

    def __init__( self, parameters={} ):
        """ Params can contain any useful parameters for the algorithm """
        self.params = {}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.min = 0
        self.max = 1
        
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        self.min = np.amin(ytrain)
        self.max = np.amax(ytrain)

    def predict(self, Xtest):
        ytest = np.random.rand(Xtest.shape[0])*(self.max-self.min) + self.min
        return ytest

class MeanPredictor(Regressor):
    """
    Returns the average target value observed; a reasonable baseline
    """
    def __init__( self, parameters={} ):
        self.params = {}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.mean = None
        
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        self.mean = np.mean(ytrain)

    def predict(self, Xtest):
        return np.ones((Xtest.shape[0],))*self.mean


class FSLinearRegression(Regressor):
    """
    Linear Regression with feature selection, and ridge regularization
    """
    def __init__( self, parameters={} ):
        self.params = {'features': [1,2,3,4,5]}
        self.reset(parameters)

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        # Dividing by numsamples before adding ridge regularization
        # to make the regularization parameter not dependent on numsamples
        numsamples = Xtrain.shape[0]
        Xless = Xtrain[:,self.params['features']]
        
        ####### Using pseudo inverse we can solve singular matrix error ######
        
        self.weights = np.dot(np.dot(np.linalg.pinv(np.dot(Xless.T,Xless)/numsamples), Xless.T),ytrain)/numsamples

    def predict(self, Xtest):
        Xless = Xtest[:,self.params['features']]
        ytest = np.dot(Xless, self.weights)
        return ytest

class RidgeLinearRegression(Regressor):
    """
    Linear Regression with ridge regularization (l2 regularization)
    TODO: currently not implemented, you must implement this method
    Stub is here to make this more clear
    Below you will also need to implement other classes for the other algorithms
    """
    def __init__( self, parameters={} ):
        # Default parameters, any of which can be overwritten by values passed to params
        self.params = {'regwgt': 0.01}
        self.reset(parameters)
        
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        # Dividing by numsamples before adding ridge regularization
        # to make the regularization parameter not dependent on numsamples
        numsamples = Xtrain.shape[0]
        # calculating weights with regularization
        self.weights = np.dot(np.dot(np.linalg.inv(np.add(np.dot(Xtrain.T,Xtrain)/numsamples,np.dot(np.identity(Xtrain.shape[1]), self.params['regwgt']))), Xtrain.T),ytrain)/numsamples

    def predict(self, Xtest):
        ytest = np.dot(Xtest, self.weights)
        return ytest
    
class LassoRegression(Regressor):
    
    """
    Linear Regression with lasso: l1 regularizer
    """
    def __init__( self, parameters={} ):
        # Default parameters, any of which can be overwritten by values passed to params
        self.params = {'regwgt': 0.01}
        self.reset(parameters)
        
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        numsamples = Xtrain.shape[0]
        self.weights = np.zeros(Xtrain.shape[1])
        error = math.inf
        tolerance = 10**-3
        maxiter = 1000
        itern = 0;
        
        # Precomputing  of matrices
        XX = np.dot(Xtrain.T,Xtrain)/numsamples
        Xy = np.dot(Xtrain.T,ytrain)/numsamples
        
        stepsize = 1/(2*np.linalg.norm(XX))
        value = stepsize*self.params['regwgt']
        #Calculating cost, divide by num samples and multiply by half for MSE
        cost = 0.5 * (np.dot(np.transpose(np.subtract(np.dot(Xtrain, self.weights), ytrain)),np.subtract(np.dot(Xtrain, self.weights), ytrain))/numsamples) + self.params['regwgt'] * np.linalg.norm(self.weights, 1)
        
        while (abs(cost-error) > tolerance) and (itern<maxiter):
            itern += 1
            error = cost
            
            self.weights = self.weights - (stepsize * np.dot(XX, self.weights)) + (stepsize * Xy)
            #proximal operator
            for i in range(Xtrain.shape[1]):
                if self.weights[i] > value:
                    self.weights[i] = self.weights[i] - value
                elif self.weights[i] < -value:
                    self.weights[i] = self.weights[i] - value
                elif self.weights[i] >= -value and self.weights[i] <= value:
                    self.weights[i] = 0
            #new cost
            cost = 0.5 * (np.dot(np.transpose(np.subtract(np.dot(Xtrain, self.weights), ytrain)),np.subtract(np.dot(Xtrain, self.weights), ytrain))/numsamples) + self.params['regwgt'] * np.linalg.norm(self.weights, 1)
            #print(cost)
            
    def predict(self, Xtest):
         ytest = np.dot(Xtest, self.weights)
         return ytest
        
class StochasticGradientDescent(Regressor):
    """
    Linear Regression through stochastic gradient descent
    """
    def __init__( self, parameters={} ):
        # Default parameters, any of which can be overwritten by values passed to params
        self.params = {'regwgt': 0.01, 'features': range(385)}
        self.reset(parameters)
        
        
    def learn(self, Xtrain, ytrain):
        self.errors = []
        self.runtime = []
        numsamples = Xtrain.shape[0]
        self.weights = np.zeros(Xtrain.shape[1])
        stepsize = .01
        epoch = 1000
        
        start_time = time.time()
        
        for i in range(epoch):
            #Shuffle the data
            dt = np.c_[Xtrain,ytrain]
            np.random.shuffle(dt)
            Xtrain = dt[:,self.params['features']]
            ytrain = dt[:,-1]
            
            for j in range(numsamples):
                # Gradient update
                gradient = np.dot(Xtrain[j].T, np.subtract(np.dot(Xtrain[j], self.weights), ytrain[j]))
                stepsize = 0.01 / (epoch + 1)
                self.weights = self.weights - (stepsize * gradient)
                cost = 0.5 * ((np.linalg.norm(np.subtract(np.dot(Xtrain[j,:], self.weights), ytrain[j])) ** 2))
                #print("cost: " + str(cost))
            self.errors.append(cost)
            
            elapsed_time = time.time() - start_time
            self.runtime.append(int(elapsed_time))
                
    def predict(self, Xtest):
        # Xless = Xtest[:,self.params['features']]
        ytest = np.dot(Xtest, self.weights)
        return ytest
    
class BatchGradientDescent(Regressor):
    """
    Linear Regression through BatchGradientDescent
    """
    def __init__( self, parameters={} ):
        # Default parameters, any of which can be overwritten by values passed to params
        self.params = {'regwgt': 0.01, 'features': range(385)}
        self.reset(parameters)
        
    def learn(self, Xtrain, ytrain):
        self.errors = []
        self.runtime = []
        numsamples = Xtrain.shape[0]
        self.weights = np.zeros(Xtrain.shape[1])
        error = math.inf
        tolerance = 10**-3
        maxiter = 1000
        itern = 0
        test = 0
        
        maxstep = 0.05
        reductionop = 0.1
        
        #cost calculation
        cost = 0.5 * ((np.linalg.norm(np.subtract(np.dot(Xtrain, self.weights), ytrain)) ** 2) / numsamples)
        start_time = time.time()
        
        while (abs(cost-error) > tolerance) and (itern<maxiter):
            itern += 1
            test += 1
            error = cost
            gradient = np.dot(np.transpose(Xtrain), np.subtract(np.dot(Xtrain, self.weights), ytrain)) / numsamples
            
            ##### Line search algorithm ###############
            step = maxstep
            obj = cost
            weightn = self.weights
            
            for i in range(maxiter):
                test+=1
                weightn = self.weights - (step * gradient)
                ncost = 0.5 * ((np.linalg.norm(np.subtract(np.dot(Xtrain, weightn), ytrain)) ** 2) / numsamples)
                if ncost < (obj - tolerance):
                    break
                else:
                    step = reductionop * step
                    
            if i == (maxiter-1):
                step = 0
                print("step" + str(i))
                
            self.weights = self.weights - (step * gradient)
            cost = 0.5 * ((np.linalg.norm(np.subtract(np.dot(Xtrain, self.weights), ytrain)) ** 2) / numsamples)
            self.errors.append(cost)
            
            elapsed_time = time.time() - start_time
            self.runtime.append(int(elapsed_time))
            #print(cost)
            
        print ("Batch Gradient Descent process input data " + str(test) + " times")
                    
            
    def predict(self, Xtest):
        # Xless = Xtest[:,self.params['features']]
        ytest = np.dot(Xtest, self.weights)
        return ytest
    
class StochasticGradientWithRMSPROP(Regressor):
    """
    Linear Regression weight optimization through RMSPROP
    """
    def __init__( self, parameters={} ):
        # Default parameters, any of which can be overwritten by values passed to params
        self.params = {'regwgt': 0.01, 'features': range(385)}
        self.reset(parameters)
        
    def learn(self, Xtrain, ytrain):
        numsamples = Xtrain.shape[0]
        self.weights = np.zeros(Xtrain.shape[1])
        stepsize = .001
        epoch = 1000
        eps_stable = 10**-8
        mean_squared_gradient = np.zeros(Xtrain.shape[1])
        
        for i in range(epoch):
            #Shuffle the data
            dt = np.c_[Xtrain,ytrain]
            np.random.shuffle(dt)
            Xtrain = dt[:,self.params['features']]
            ytrain = dt[:,-1]
            
            for j in range(numsamples):
                # Gradient update
                gradient = np.dot(Xtrain[j].T, np.subtract(np.dot(Xtrain[j], self.weights), ytrain[j]))
                
                ### RMSPROP implementation#########
                
                mean_squared_gradient = 0.9 * mean_squared_gradient + 0.1 * gradient ** 2
                self.weights -= ((stepsize * gradient) / np.sqrt(mean_squared_gradient + eps_stable))
                gradient = np.dot(np.transpose(Xtrain), np.subtract(np.dot(Xtrain, self.weights), ytrain)) / numsamples
                
            cost = 0.5 * ((np.linalg.norm(np.subtract(np.dot(Xtrain, self.weights), ytrain)) ** 2) / numsamples)    
            #print (cost)
        
    def predict(self, Xtest):
        # Xless = Xtest[:,self.params['features']]
        ytest = np.dot(Xtest, self.weights)
        return ytest
    
class StochasticGradientWithAMSGRAD(Regressor):
    """
    Linear Regression weight optimization AMSGRAD
    """
    def __init__( self, parameters={} ):
        # Default parameters, any of which can be overwritten by values passed to params
        self.params = {'regwgt': 0.01, 'features': range(385)}
        self.reset(parameters)
        
    def learn(self, Xtrain, ytrain):
        numsamples = Xtrain.shape[0]
        self.weights = np.zeros(Xtrain.shape[1])
        stepsize = .001
        epoch = 1000
        eps_stable = 10**-8
        beta_1 = 0.9
        beta_2 = 0.999
        m_t = np.zeros(Xtrain.shape[1])
        v_t = np.zeros(Xtrain.shape[1])
        v_cap = np.zeros(Xtrain.shape[1])
       
        for i in range(epoch):
            #Shuffle the data
            dt = np.c_[Xtrain,ytrain]
            np.random.shuffle(dt)
            Xtrain = dt[:,self.params['features']]
            ytrain = dt[:,-1]
            
            for j in range(numsamples):
                # Gradient update
                gradient = np.dot(Xtrain[j].T, np.subtract(np.dot(Xtrain[j], self.weights), ytrain[j]))
                
                ### AMSGRAD implementation#########
                
                m_t = beta_1*m_t + (1-beta_1)*gradient	
                v_t = beta_2*v_t + (1-beta_2)*(gradient**2)
                v_cap = np.maximum(v_cap,v_t)	
                self.weights -= ((stepsize*m_t)/(np.sqrt(v_cap)+eps_stable))	#update the weights
                
            cost = 0.5 * ((np.linalg.norm(np.subtract(np.dot(Xtrain, self.weights), ytrain)) ** 2) / numsamples)    
            #print (cost)
        
    def predict(self, Xtest):
        # Xless = Xtest[:,self.params['features']]
        ytest = np.dot(Xtest, self.weights)
        return ytest