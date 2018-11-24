from __future__ import division  # floating point division
import csv
import random
import math
import numpy as np
from matplotlib import pyplot as plt

import dataloader as dtl
import regressionalgorithms as algs
import utilities as utl

def l2err(prediction,ytest):
    """ l2 error (i.e., root-mean-squared-error) """
    return np.linalg.norm(np.subtract(prediction,ytest))

def l1err(prediction,ytest):
    """ l1 error """
    return np.linalg.norm(np.subtract(prediction,ytest),ord=1)

def l2err_squared(prediction,ytest):
    """ l2 error squared """
    return np.square(np.linalg.norm(np.subtract(prediction,ytest)))

def geterror(predictions, ytest):
    # Can change this to other error values
    return 0.5*l2err_squared(predictions,ytest)/ytest.shape[0]


if __name__ == '__main__':
    trainsize = 1000
    testsize = 5000
    numruns = 2
    regressionalgs = {#'Random': algs.Regressor(),
                #'Mean': algs.MeanPredictor(),
                #'FSLinearRegression5': algs.FSLinearRegression({'features': [1,2,3,4,5]}),
                #'FSLinearRegression50': algs.FSLinearRegression({'features': range(50)}),
                'FSLinearRegression385': algs.FSLinearRegression({'features': range(385)}),
                'RidgeLinearRegression': algs.RidgeLinearRegression(),
                'LassoRegression': algs.LassoRegression(),
                'StochasticGradientDescent': algs.StochasticGradientDescent(),
                'BatchGradientDescent': algs.BatchGradientDescent(),
                'StochasticGradientDescentWithRMSPROP': algs.StochasticGradientWithRMSPROP(),
                'StochasticGradientDescentWithAMSGRAD': algs.StochasticGradientWithAMSGRAD()
             }
    numalgs = len(regressionalgs)

    # Enable the best parameter to be selected, to enable comparison
    # between algorithms with their best parameter settings
    parameters = (
        #{'regwgt': 0.0},
        {'regwgt': 0.01, 'features': range(385)},
        #{'regwgt': 1.0},
                      )
    numparams = len(parameters)
    allerrorSGD = []
    allerrorBGD = []
    allruntimeSGD = []
    allruntimeBGD = []
    errors = {}
    
    for learnername in regressionalgs:
        errors[learnername] = np.zeros((numparams,numruns))

    for r in range(numruns):
        trainset, testset = dtl.load_ctscan(trainsize,testsize)
        print(('Running on train={0} and test={1} samples for run {2}').format(trainset[0].shape[0], testset[0].shape[0],r))

        for p in range(numparams):
            params = parameters[p]
            for learnername, learner in regressionalgs.items():
                # Reset learner for new parameters
                learner.reset(params)
                print ('Running learner = ' + learnername + ' on parameters ' + str(learner.getparams()))
                # Train model
                learner.learn(trainset[0], trainset[1])
                # Test model
                predictions = learner.predict(testset[0])
                error = geterror(testset[1], predictions)
                print ('Error for ' + learnername + ': ' + str(error))
                errors[learnername][p,r] = error
                
                ## saving error for graph
                if learnername == "StochasticGradientDescent":
                    allerrorSGD.append(learner.errors)
                    allruntimeSGD.append(learner.runtime)
                elif learnername == "BatchGradientDescent":
                    allerrorBGD.append(learner.errors)
                    allruntimeBGD.append(learner.runtime)
                    

    for learnername in regressionalgs:
        besterror = np.mean(errors[learnername][0,:])
        ## Standard Error
        beststandarderror = utl.stdev(errors[learnername][0,:])/math.sqrt(numruns)
        bestparams = 0
        for p in range(numparams):
            aveerror = np.mean(errors[learnername][p,:])
            ## Standard Error
            standarderror = utl.stdev(errors[learnername][p,:])/math.sqrt(numruns)
            if aveerror < besterror:
                besterror = aveerror
                beststandarderror = standarderror
                bestparams = p

        # Extract best parameters
        learner.reset(parameters[bestparams])
        #print ('Best parameters for ' + learnername + ': ' + str(learner.getparams()))
        print ('Average error for ' + learnername + ': ' + str(besterror))
        print ('Standard error for ' + learnername + ': ' + str(beststandarderror))

####### GRAPH for SGD and BGD ###############
        #### Cost vs Epochs

    costSGD = [sum(i)/len(i) for i in zip(*allerrorSGD)]
    costBGD = [sum(i)/len(i) for i in zip(*allerrorBGD)]

    print ("Average cost vs iterations for SGD vs BGD")
    plt.ylabel("Error")
    plt.xlabel("Epochs")
    plt.axis([0, 1000, 0, 800])
    plt.plot(costSGD, 'g-', label="SGD")
    plt.plot(costBGD, 'b-', label="BGD")
    plt.legend(loc='upper left')
    plt.show()
    
    #### Cost vs Run time SGD

    runtimeSGD = [sum(i)/len(i) for i in zip(*allruntimeSGD)]

    print ("Average cost vs runtime for SGD")
    plt.ylabel("Error")
    plt.xlabel("Runtime")
    plt.axis([0, 30, 0, 800])
    plt.plot(runtimeSGD, costSGD, 'g-', label="SGD")
    plt.show()
    
     #### Cost vs Run time SGD

    runtimeBGD = [sum(i)/len(i) for i in zip(*allruntimeSGD)]

    print ("Average cost vs runtime for BGD")
    plt.ylabel("Error")
    plt.xlabel("Runtime")
    plt.axis([0, 30, 0, 800])
    plt.plot(runtimeBGD, costBGD, 'b-', label="BGD")
    plt.show()