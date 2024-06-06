# -*- coding: utf-8 -*-
"""
Created on Tue May 29 20:33:15 2023

@author: Emre Can Şen
"""

import numpy as np
from skmultiflow.drift_detection import ADWIN
from skmultiflow.data.sea_generator import SEAGenerator
from skmultiflow.data.agrawal_generator import AGRAWALGenerator
from skmultiflow.evaluation import EvaluatePrequential
from skmultiflow.meta import DynamicWeightedMajorityClassifier
from skmultiflow.trees import HoeffdingTreeClassifier
from skmultiflow.meta import AdaptiveRandomForestClassifier
from skmultiflow.lazy import SAMKNNClassifier
from skmultiflow.meta import StreamingRandomPatchesClassifier
from skmultiflow.data.concept_drift_stream import ConceptDriftStream
from skmultiflow.data.file_stream import FileStream
import matplotlib.pyplot as plt


# Ensemble with drift Hoeffding handling
class HoeffdingEnsemble:
    def __init__(self, classifier_num=5):
        self.classifier_num = classifier_num
        self.classifiers = [HoeffdingTreeClassifier() for _ in range(classifier_num)]
        self.drift_detectors = [ADWIN() for _ in range(classifier_num)]

    def partial_fit(self, X, y, classes=None):
        for i in range(self.classifier_num):
            self.classifiers[i].partial_fit(X, y, classes)
            self.drift_detectors[i].add_element(y[0] - self.classifiers[i].predict(X)[0])

            # Drift handling
            if self.drift_detectors[i].detected_change():
                self.classifiers[i] = HoeffdingTreeClassifier()
                self.drift_detectors[i] = ADWIN()

    def predict(self, X):
        prediction = [self.classifiers[i].predict(X) for i in range(self.classifier_num)]
        return np.round(np.mean(prediction, axis=0))
    
# Synthetic data generation
agrawal_generator = AGRAWALGenerator()
sea_generator = SEAGenerator()
  
# Initialize Drift
agrawal_generator_drift= ConceptDriftStream(stream=agrawal_generator, drift_stream=agrawal_generator,
                                         position=10000,
                                          width=10000)

sea_generator_drift = ConceptDriftStream(stream=sea_generator, drift_stream=sea_generator,
                                          position=20000,
 
                                          width=10000)
# Prepare data for classifier use
agrawal_generator_drift.prepare_for_use()
sea_generator_drift.prepare_for_use()
# Generate 100,000 data instances with AGRAWALGenerator 
agrawal_data = []
for _ in range(100000):
    a, b = agrawal_generator_drift.next_sample()
    agrawal_data.append((a, b))
    
# Generate 100,000 data instances with SEAGenerator   
sea_data = []
for _ in range(100000):
    a, b = sea_generator_drift.next_sample()
    sea_data.append((a, b))
    
# Save the AGRAWALGenerator data to a file
with open('AGRAWALGenerator.csv', 'w') as f:
    for data in agrawal_data:
        a, b = data
        line = ','.join(str(feature) for feature in a) + ',' + str(b) + '\n'
        f.write(line)    
  
# Save the SEAGenerator data to a file
with open('SEADataset.csv', 'w') as f:
    for data in sea_data:
        a, b = data
        line = ','.join(str(feature) for feature in a) + ',' + str(b) + '\n'
        f.write(line)
        
AGRAWALDataset = FileStream("C:/Users/Emre/Documents/ge461_project4/AGRAWALGenerator()erator.csv")

#Load Data
spam_data = FileStream("C:/Users/Emre/Documents/ge461_project4/spam.csv")
electricity_data = FileStream("C:/Users/Emre/Documents/ge461_project4/elec.csv")

# Initialize Classifiers
arf_classifier = AdaptiveRandomForestClassifier()
samknn_classifier = SAMKNNClassifier()
srp_classifier = StreamingRandomPatchesClassifier(n_estimators=3)
dwm_classifier = DynamicWeightedMajorityClassifier()
hoeff_ensemble = HoeffdingEnsemble(classifier_num=10)

# Initialize Evaluator
evaluator_arf = EvaluatePrequential(show_plot=True, metrics=["accuracy"])
evaluator_sam_knn = EvaluatePrequential(show_plot=True, metrics=["accuracy"])
evaluator_srp = EvaluatePrequential(show_plot=True, metrics=["accuracy"])
evaluator_dwm = EvaluatePrequential(show_plot=True, metrics=["accuracy"])
evaluator_ensemble = EvaluatePrequential(show_plot=True, metrics=["accuracy"])

# Prequential Accuracy Plot
#--This code is only used for ARGWAL dataset, afterwards only sliding window code is utilized and this part is commented out
evaluator_arf.evaluate(stream=agrawal_generator_drift, model=arf_classifier, model_names=['ARF'])
evaluator_sam_knn.evaluate(stream=agrawal_generator_drift, model=samknn_classifier, model_names=['SAM_kNN'])
evaluator_srp.evaluate(stream=agrawal_generator_drift, model=srp_classifier, model_names=['SRP'])
evaluator_dwm.evaluate(stream=agrawal_generator_drift, model=dwm_classifier, model_names=['DWM'])
evaluator_ensemble.evaluate(stream=spam_data, model=hoeff_ensemble, model_names=['Ensemble'])


# Create a list of classifiers for iteration
# The variable names and plot names are entered below and iterated for all classifiers and datasets
classifiers = [srp_classifier]
names = ['SRP Classifier- SEA Dataset']

#Sliding windows for prequential accuracy
num_windows = 20
window_size = 100000 // num_windows

# Accuracy lists for results
accuracies = []
prequential_accuracies = [[] for _ in range(len(classifiers))]

# Above there are two parts one for generating real and one for generating the synthetic prequential accuracy plots 
# One is commented out while the other is running  

# #Sliding Windows for real datasets 
for stream in [electricity_data]:
    stream.restart()  
    n_samples = stream.n_remaining_samples()
    window_size = n_samples // num_windows  
    # Iterate through each classifier
    for x, c in enumerate(classifiers):
        correct = 0
        window_correct = 0
        # Train and test using Interleaved Test-Then-Train approach
        for i in range(n_samples):
            a, b = stream.next_sample()
            y_prediction = c.predict(a)
            if b == y_prediction:
                correct += 1
                window_correct += 1
            c.partial_fit(a, b)
            # Calculate prequential accuracy for the current window
            if (i+1) % window_size == 0:
                prequential_accuracies[x].append(window_correct / window_size)
                window_correct = 0
        # Overall accuracy
        accuracies.append(correct/n_samples)
    # Print overall accuracies
    for name, acc in zip(names, accuracies):
        print(f"Overall accuracy of {name}: {acc:.4f}")
    # Plot prequential accuracies
    plt.figure(figsize=(10, 6))
    for x, name in enumerate(names):
        plt.plot(prequential_accuracies[x], label=name)

    plt.xlabel('Window Number')
    plt.ylabel('Prequential Accuracy')
    plt.legend()
    plt.show()


# #Sliding Windows for synthetic datasets 
for x, c in enumerate(classifiers):
    correct = 0
    window_correct = 0
    # Train and test using Interleaved Test-Then-Train approach
    sea_generator_drift.restart()
    for i in range(100000):
        a, b = sea_generator_drift.next_sample()
        y_prediction = c.predict(a)
        if b == y_prediction:
            correct += 1
            window_correct += 1
        c.partial_fit(a, b)
        if i % window_size == 0 and i > 0:
            prequential_accuracies[x].append(window_correct/window_size)
            window_correct = 0
    # Calculate overall accuracy
    accuracies.append(correct/100000)

# Print overall accuracies
for name, acc in zip(names, accuracies):
    print(f"Overall accuracy of {name}: {acc:.4f}")

# Plot prequential accuracies
plt.figure(figsize=(10, 6))
for x, name in enumerate(names):
    plt.plot(prequential_accuracies[x], label=name)

plt.xlabel('Window number')
plt.ylabel('Prequential accuracy')
plt.legend()
plt.show()