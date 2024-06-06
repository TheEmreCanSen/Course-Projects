# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 03:25:51 2022

@author: Emre
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data = pd.read_csv("dataset.csv")

# y_sk = data.label.values
# x_data_sk = data.drop(["label"],axis=1)
# x_sk= (x_data_sk - np.min(x_data_sk)) / (np.max(x_data_sk) - np.min(x_data_sk)).values

# x_train_sk, x_test_sk, y_train_sk, y_test_sk = train_test_split(x_sk,y_sk,test_size = 0.2, random_state = 42)

# x_train_sk = x_train_sk.T
# x_test_sk = x_test_sk.T
# y_train_sk = y_train_sk.T
# y_test_sk = y_test_sk.T

x_data = data.drop(["label"],axis=1)
y = data.label.values
x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values
x['label']=y.tolist()
data=x

data = data.sample(frac = 1, random_state=42)

train_size = 0.7
valid_size=0.2

train_index = int(len(data)*train_size)

data_train = data[0:train_index]
data_rem = data[train_index:]

valid_index = int(len(data)*valid_size)

data_valid = data[train_index:train_index+valid_index]
data_test = data[train_index+valid_index:]

x_train_wosk, y_train_wosk = data_train.drop(columns='label').copy(), data_train['label'].copy(),
x_valid_wosk, y_valid_wosk = data_valid.drop(columns='label').copy(), data_valid['label'].copy()
x_test_wosk, y_test_wosk = data_test.drop(columns='label').copy(), data_test['label'].copy()

x_train_old=x_train_wosk
y_train_old=y_train_wosk

x_train_wosk=pd.concat([x_train_wosk, x_valid_wosk])
x_train_wosk=x_train_wosk.T
x_test_wosk = x_test_wosk.T
y_train_wosk=pd.concat([y_train_wosk, y_valid_wosk])
y_train_wosk=y_train_wosk.T
y_train_wosk=y_train_wosk.to_numpy(dtype="int64")
y_test_wosk = y_test_wosk.T
y_test_wosk=y_test_wosk.to_numpy(dtype="int64")

def sigmoid(z):
    
    y_hold = 1 / (1+np.exp(-z))
    
    return y_hold

def weights_and_bias(dimension):
    
    w = np.random.normal(loc=0.0, scale=1.0, size=(dimension, 1))  
    b = 0.0
    return w,b

def full_batch(w,bias,x_train,y_train):
    
    z = np.dot(w.T,x_train) + bias
    y_hold = sigmoid(z)
    loss = (-y_train)*np.log(y_hold) - ((1-y_train))*np.log(1-y_hold)
    cost = (np.sum(loss)) / x_train.shape[1]
    
    #backward propogation
    weight = (np.dot(x_train,((y_hold-y_train).T)))/x_train.shape[1]
    derivative_bias = np.sum(y_hold-y_train)/x_train.shape[1]
    gradients = {"weight": weight,"derivative_bias": derivative_bias}
    return cost,gradients

def predict(w,bias,x_test):
    # x_test is a input for forward propagation
    z = sigmoid(np.dot(w.T,x_test)+bias)
    Y_prediction = np.zeros((1,x_test.shape[1]))
    # if z is bigger than 0.5, our prediction is one means has diabete (y_hold=1),
    # if z is smaller than 0.5, our prediction is zero means does not have diabete (y_hold=0),
    for i in range(z.shape[1]):
        if z[0,i]<= 0.5:
            Y_prediction[0,i] = 0
        else:
            Y_prediction[0,i] = 1

    return Y_prediction

def update(w, bias, x_train, y_train, learning_rate,number_of_iterarion):
    cost_l = []
    cost_l_two = []
    index = []
    # updating(learning) parameters is number_of_iterarion times
    for i in range(number_of_iterarion):
        # make forward and backward propagation and find cost and gradients
        cost,gradients = full_batch(w,bias,x_train,y_train)
        cost_l.append(cost)
        # lets update
        w = w - learning_rate * gradients["weight"]
        bias = bias - learning_rate * gradients["derivative_bias"]
        if i % 10 == 0:
            cost_l_two.append(cost)
            index.append(i)
            print ("Cost after iteration %i: %f" %(i, cost)) #if section defined to print our cost values in every 10 iteration. We do not need to do that. It's optional.
    # we update(learn) parameters weights and bias
    parameters = {"weight": w,"bias": bias}
    plt.plot(index,cost_l_two)
    plt.xticks(index,rotation='vertical')
    plt.xlabel("Number of Iteration")
    plt.ylabel("Cost")
    plt.show()
    return parameters, gradients, cost_l


def logistic_regression(x_train, y_train, x_test, y_test, learning_rate ,  num_iterations):
    # initialize
    dimension =  x_train.shape[0]
    w,bias = weights_and_bias(dimension)
    
    parameters, gradients, cost_l = update(w, bias, x_train, y_train, learning_rate,num_iterations)
    
    y_prediction_test = predict(parameters["weight"],parameters["bias"],x_test)
    

    # Print train/test Errors
    
    print("Test acc.: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))
    
    
    
logistic_regression(x_train_wosk, y_train_wosk, x_test_wosk, y_test_wosk,learning_rate = 0.01, num_iterations = 100)



