# -*- coding: utf-8 -*-
"""
Created on Thu May 18 15:46:12 2023

@author: Emre
"""

import matplotlib.pyplot as plot
import numpy as np
import random

f = open('./test1.txt', 'r')
inputs = []
outputs = []

for i in f:
    i = i.strip().split('\t')
    inputs.append(i[0])
    outputs.append(i[1])

inputsTest = np.array(inputs, dtype=float)
outputsTest = np.array(outputs, dtype=float)
f.close()

f = open('./train1.txt', 'r')
inputs = []
outputs = []

for i in f:
    i = i.strip().split('\t')
    inputs.append(i[0])
    outputs.append(i[1])

inputsTrain = np.array(inputs, dtype=float)
outputsTrain = np.array(outputs, dtype=float)

inputsTrain_norm = (inputsTrain-np.min(inputsTrain))/(np.max(inputsTrain)-np.min(inputsTrain))
outputsTrain_norm = (outputsTrain-np.min(outputsTrain))/(np.max(outputsTrain)-np.min(outputsTrain))

class ArtificialNeuralNetwork(object):
    def __init__(self, hiddenLayer):
        self.hiddenLayer = hiddenLayer
        self.hiddenLayerWeight = np.array([random.random() for i in range(hiddenLayer)]) 
        self.inputWeight = np.array([random.random() for i in range(hiddenLayer)]) 
        self.outputWeight = 1 / float(hiddenLayer)
    def sigmoid(self,x):
        return 1 / (1 + np.exp(-(x)))
    
    def derive_sig(self, sigm): 
        return sigm * (1-sigm)

    def error_calc(self, out, sum):
        return out - sum

    def create_plt(self, input, output):
        pred = self.prediction
        totalLose = 0
        totalLose = sum(pow((pred - output), 2)) 

        plot.scatter(input, output, label = "Data points") 
        plot.scatter(input, pred, label = "Predicted") 
        plot.xlabel('Inputs')
        plot.ylabel('Outputs')
        plot.legend()
        plot.title("ANN Squared Error= " + str(totalLose))
        plot.show()

    def train(self, input, output, epoch, lr):
        for i in range (epoch):
           
            index = np.random.randint(0,len(input))   
            hiddenValue = self.hiddenLayerWeight
            inputValue = self.inputWeight
            outputValue = self.outputWeight

            fx = inputValue + input[index] * hiddenValue 
            fxSigmoid = self.sigmoid(fx)
            fxSigmoidDerivative = self.derive_sig(fxSigmoid)
            errorF = self.error_calc(output[index], np.sum(fxSigmoid * outputValue))
            
            self.inputWeight = self.inputWeight + (lr * errorF * outputValue * fxSigmoidDerivative)
            self.outputWeight = self.outputWeight + (lr * errorF * fxSigmoid)
            self.hiddenLayerWeight = self.hiddenLayerWeight + (lr * errorF * outputValue * fxSigmoidDerivative * input[index])

            fxWithReshape = inputValue + input.reshape(len(input), 1) * hiddenValue
            fxSigmoid = self.sigmoid(fxWithReshape)
            prediction = np.dot(fxSigmoid, outputValue)
            totalLose = 0
            totalLose = sum(pow((prediction - output.reshape(len(output),1)), 2))

    def predict(self, input, output, tp, tpValue):
        hiddenValue = self.hiddenLayerWeight
        inputValue = self.inputWeight
        outputValue = self.outputWeight
        hiddenLayerC = self.hiddenLayer
        fxWithReshape = inputValue + input.reshape(len(input), 1) * hiddenValue
        fxSigmoid = self.sigmoid(fxWithReshape)
        self.prediction = np.dot(fxSigmoid, outputValue)
        pred = self.prediction
        totalLose = 0
        totalLose = sum(pow((pred - output), 2)) 
        print("For " + str(tpValue), tp, ", total loss =", totalLose)

    def calculate_res(self, input, output):
        hiddenValue = self.hiddenLayerWeight
        inputValue = self.inputWeight
        outputValue = self.outputWeight
        hiddenLayerC = self.hiddenLayer
        fxWithReshape = inputValue + input.reshape(len(input), 1) * hiddenValue
        fxSigmoid = self.sigmoid(fxWithReshape)
        self.prediction = np.dot(fxSigmoid, outputValue)
        pred = self.prediction
        totalLose = 0
        averageLoss = 0
        standartDerivation = 0
        totalLose = sum(pow((pred - output), 2)) 

        averageLoss = totalLose / len(input)
        standartDerivation = sum(pow((input - averageLoss),2))
        standartDerivation = np.sqrt(standartDerivation / (len(input)-1))
        return averageLoss, standartDerivation

epochs = 10000
lr = 0.001
hiddenUnits = 2
tp = 'hidden units'


for i in range(6):
    annModel = ArtificialNeuralNetwork(hiddenUnits)
    annModel.train(inputsTrain, outputsTrain, epochs, lr)
    annModel.predict(inputsTrain, outputsTrain, tp, hiddenUnits)
    hiddenUnits = hiddenUnits * 2
print("################################")

bestUnit = 64
lr = 0.01
tp = 'learning rate'

for i in range(4):
    annModel = ArtificialNeuralNetwork(bestUnit)
    annModel.train(inputsTrain, outputsTrain, epochs, lr)
    annModel.predict(inputsTrain, outputsTrain, tp, float(lr))
    lr = lr / 10

print("################################")

bestLr = 0.001
epochs = 10
tp = 'epoch'

for i in range(5):
    annModel = ArtificialNeuralNetwork(bestUnit)
    annModel.train(inputsTrain, outputsTrain, epochs, bestLr)
    annModel.predict(inputsTrain, outputsTrain, tp, float(epochs))
    epochs = epochs * 10

print("################################")

print("WITH BEST VARIABLES")
bestEpoch = 100000 
annModel = ArtificialNeuralNetwork(bestUnit)
annModel.train(inputsTrain, outputsTrain, bestEpoch, bestLr)
annModel.predict(inputsTrain, outputsTrain, tp, float(bestEpoch))
annModel.create_plt(inputsTrain, outputsTrain)

averageLoss, standartDerivation = annModel.calculate_res(inputsTrain, outputsTrain)
print("Average loss for train with ", bestUnit, " unit, ", bestEpoch, " epoch, ", bestLr, " best learning rate, ", "Average loss =", averageLoss)
print("Standart Derivation  for train with ", bestUnit, " unit, ", bestEpoch, " epoch, ", bestLr, " best learning rate, ", "Standart Derivation =", standartDerivation)

bestEpoch = 100000 
annModel = ArtificialNeuralNetwork(bestUnit)
annModel.train(inputsTest, outputsTest, bestEpoch, bestLr)
annModel.predict(inputsTest, outputsTest, tp, float(bestEpoch))
annModel.create_plt(inputsTest, outputsTest)

averageLoss, standartDerivation = annModel.calculate_res(inputsTest, outputsTest)
print("Average loss for test with ", bestUnit, " unit, ", bestEpoch, " epoch, ", bestLr, " best learning rate, ", "Average loss =", averageLoss)


########PART C###############

print("BEST LEARNING RATE AND EPOCH FOR EACH OF THE HIDDEN UNIT NUMBER CONFIGURATIONS(TRAIN DATA)")
tp = 'hidden units'
hiddenUnits = 2
epochs = 10000
lr = 0.001
annModel = ArtificialNeuralNetwork(hiddenUnits)
annModel.train(inputsTrain, outputsTrain, epochs, lr)
annModel.predict(inputsTrain, outputsTrain, tp, hiddenUnits)
annModel.create_plt(inputsTrain, outputsTrain)
averageLoss, standartDerivation = annModel.calculate_res(inputsTrain, outputsTrain)
print("Average loss for train with ", hiddenUnits, " unit, ", epochs, " epoch, ", lr, " best learning rate, ", "Average loss =", averageLoss)
print("Standart Derivation  for train with ", hiddenUnits, " unit, ", epochs, " epoch, ", lr, " best learning rate, ", "Standart Derivation =", standartDerivation)
print("################################")


print("BEST LEARNING RATE AND EPOCH FOR EACH OF THE HIDDEN UNIT NUMBER CONFIGURATIONS(TEST DATA)")
tp = 'hidden units'
hiddenUnits = 2
epochs = 10000
lr = 0.001
annModel = ArtificialNeuralNetwork(hiddenUnits)
annModel.train(inputsTest, outputsTest, epochs, lr)
annModel.predict(inputsTest, outputsTest, tp, hiddenUnits)
annModel.create_plt(inputsTest, outputsTest)
averageLoss, standartDerivation = annModel.calculate_res(inputsTest, outputsTest)
print("Average loss for test with ", hiddenUnits, " unit, ", epochs, " epoch, ", lr, " best learning rate, ", "Average loss =", averageLoss)
print("Standart Deviation  for test with ", hiddenUnits, " unit, ", epochs, " epoch, ", lr, " best learning rate, ", "Standart Deviation =", standartDerivation)
print("################################")

print("HIDDEN UNITS:4 (TRAIN DATA)")
tp = 'hidden units'
hiddenUnits = 4
epochs = 100000
lr = 0.001
annModel = ArtificialNeuralNetwork(hiddenUnits)
annModel.train(inputsTrain, outputsTrain, epochs, lr)
annModel.predict(inputsTrain, outputsTrain, tp, hiddenUnits)
annModel.create_plt(inputsTrain, outputsTrain)
averageLoss, standartDerivation = annModel.calculate_res(inputsTrain, outputsTrain)
print("Average loss for train with ", hiddenUnits, " unit, ", epochs, " epoch, ", lr, " best learning rate, ", "Average loss =", averageLoss)
print("Standart Derivation  for train with ", hiddenUnits, " unit, ", epochs, " epoch, ", lr, " best learning rate, ", "Standart Derivation =", standartDerivation)
print("################################")


print("HIDDEN UNITS:4 (TEST DATA)")
tp = 'hidden units'
hiddenUnits = 4
epochs = 100000
lr = 0.001
annModel = ArtificialNeuralNetwork(hiddenUnits)
annModel.train(inputsTest, outputsTest, epochs, lr)
annModel.predict(inputsTest, outputsTest, tp, hiddenUnits)
annModel.create_plt(inputsTest, outputsTest)
averageLoss, standartDerivation = annModel.calculate_res(inputsTest, outputsTest)
print("Average loss for test with ", hiddenUnits, " unit, ", epochs, " epoch, ", lr, " best learning rate, ", "Average loss =", averageLoss)
print("Standart Deviation  for test with ", hiddenUnits, " unit, ", epochs, " epoch, ", lr, " best learning rate, ", "Standart Deviation =", standartDerivation)
print("################################")

print("HIDDEN UNITS:8 (TRAIN DATA)")
tp = 'hidden units'
hiddenUnits = 8
epochs = 100000
lr = 0.001
annModel = ArtificialNeuralNetwork(hiddenUnits)
annModel.train(inputsTrain, outputsTrain, epochs, lr)
annModel.predict(inputsTrain, outputsTrain, tp, hiddenUnits)
annModel.create_plt(inputsTrain, outputsTrain)
averageLoss, standartDerivation = annModel.calculate_res(inputsTrain, outputsTrain)
print("Average loss for train with ", hiddenUnits, " unit, ", epochs, " epoch, ", lr, " best learning rate, ", "Average loss =", averageLoss)
print("Standart Derivation  for train with ", hiddenUnits, " unit, ", epochs, " epoch, ", lr, " best learning rate, ", "Standart Derivation =", standartDerivation)
print("################################")


print("HIDDEN UNITS:8 (TEST DATA)")
tp = 'hidden units'
hiddenUnits = 8
epochs = 100000
lr = 0.001
annModel = ArtificialNeuralNetwork(hiddenUnits)
annModel.train(inputsTest, outputsTest, epochs, lr)
annModel.predict(inputsTest, outputsTest, tp, hiddenUnits)
annModel.create_plt(inputsTest, outputsTest)
averageLoss, standartDerivation = annModel.calculate_res(inputsTest, outputsTest)
print("Average loss for test with ", hiddenUnits, " unit, ", epochs, " epoch, ", lr, " best learning rate, ", "Average loss =", averageLoss)
print("Standart Deviation  for test with ", hiddenUnits, " unit, ", epochs, " epoch, ", lr, " best learning rate, ", "Standart Deviation =", standartDerivation)
print("################################")

print("HIDDEN UNITS:16 (TRAIN DATA)")
tp = 'hidden units'
hiddenUnits = 16
epochs = 100000
lr = 0.001
annModel = ArtificialNeuralNetwork(hiddenUnits)
annModel.train(inputsTrain, outputsTrain, epochs, lr)
annModel.predict(inputsTrain, outputsTrain, tp, hiddenUnits)
annModel.create_plt(inputsTrain, outputsTrain)
averageLoss, standartDerivation = annModel.calculate_res(inputsTrain, outputsTrain)
print("Average loss for train with ", hiddenUnits, " unit, ", epochs, " epoch, ", lr, " best learning rate, ", "Average loss =", averageLoss)
print("Standart Derivation  for train with ", hiddenUnits, " unit, ", epochs, " epoch, ", lr, " best learning rate, ", "Standart Derivation =", standartDerivation)
print("################################")


print("HIDDEN UNITS:16 (TEST DATA)")
tp = 'hidden units'
hiddenUnits = 16
epochs = 100000
lr = 0.001
annModel = ArtificialNeuralNetwork(hiddenUnits)
annModel.train(inputsTest, outputsTest, epochs, lr)
annModel.predict(inputsTest, outputsTest, tp, hiddenUnits)
annModel.create_plt(inputsTest, outputsTest)
averageLoss, standartDerivation = annModel.calculate_res(inputsTest, outputsTest)
print("Average loss for test with ", hiddenUnits, " unit, ", epochs, " epoch, ", lr, " best learning rate, ", "Average loss =", averageLoss)
print("Standart Deviation  for test with ", hiddenUnits, " unit, ", epochs, " epoch, ", lr, " best learning rate, ", "Standart Deviation =", standartDerivation)
print("################################")

print("HIDDEN UNITS:32 (TRAIN DATA)")
tp = 'hidden units'
hiddenUnits = 32
epochs = 100000
lr = 0.001
annModel = ArtificialNeuralNetwork(hiddenUnits)
annModel.train(inputsTrain, outputsTrain, epochs, lr)
annModel.predict(inputsTrain, outputsTrain, tp, hiddenUnits)
annModel.create_plt(inputsTrain, outputsTrain)
averageLoss, standartDerivation = annModel.calculate_res(inputsTrain, outputsTrain)
print("Average loss for train with ", hiddenUnits, " unit, ", epochs, " epoch, ", lr, " best learning rate, ", "Average loss =", averageLoss)
print("Standart Derivation  for train with ", hiddenUnits, " unit, ", epochs, " epoch, ", lr, " best learning rate, ", "Standart Derivation =", standartDerivation)
print("################################")


print("HIDDEN UNITS:32 (TEST DATA)")
tp = 'hidden units'
hiddenUnits = 32
epochs = 100000
lr = 0.001
annModel = ArtificialNeuralNetwork(hiddenUnits)
annModel.train(inputsTest, outputsTest, epochs, lr)
annModel.predict(inputsTest, outputsTest, tp, hiddenUnits)
annModel.create_plt(inputsTest, outputsTest)
averageLoss, standartDerivation = annModel.calculate_res(inputsTest, outputsTest)
print("Average loss for test with ", hiddenUnits, " unit, ", epochs, " epoch, ", lr, " best learning rate, ", "Average loss =", averageLoss)
print("Standart Deviation  for test with ", hiddenUnits, " unit, ", epochs, " epoch, ", lr, " best learning rate, ", "Standart Deviation =", standartDerivation)
print("################################")