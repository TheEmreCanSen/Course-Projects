# -*- coding: utf-8 -*-
"""
Created on Thu May 18 15:50:05 2023

@author: Emre
"""

import matplotlib.pyplot as plot
import numpy as np

f = open('./test1.txt', 'r')
inputs = []
outputs = []

for i in f:
    i = i.strip().split('\t')
    inputs.append(i[0])
    outputs.append(i[1])

f = open('./train1.txt', 'r')
inputs = []
outputs = []

for i in f:
    i = i.strip().split('\t')
    inputs.append(i[0])
    outputs.append(i[1])

inputs_train = np.array(inputs, dtype=float)
outputs_train = np.array(outputs, dtype=float)
inputSize = len(inputs_train)

inputMean = np.mean(inputs_train)
outputMean = np.mean(outputs_train)

sumX = np.sum(pow(inputs_train,2))
sumY = np.sum(outputs_train * inputs_train)

deviationOfXX = sumX - (inputSize * inputMean * inputMean)
deviationOfYX = sumY - (inputSize * inputMean * outputMean)

b0 = deviationOfYX / deviationOfXX
b1 = outputMean - ((deviationOfYX / deviationOfXX) * inputMean)

prediction = b0 * inputs_train + b1
totalLose = sum(pow((prediction - outputs_train), 2)) 

plot.scatter(inputs_train, outputs_train, label = "Data points") #points
plot.plot(inputs_train, prediction, label = "Predicted") #prediction line
plot.xlabel('Inputs')
plot.ylabel('Outputs')
plot.legend()

plot.title("Data Points vs. Prediction")
plot.show()

print("Total linear regression loss =  " + str(totalLose))