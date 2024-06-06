# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 17:23:46 2022

@author: Emre
"""

import matplotlib.pyplot as pp
import pandas as pd
import numpy as np

train_data = pd.read_csv("bbcsports_train.csv")
val_data = pd.read_csv("bbcsports_val.csv")
train_replaced = train_data.drop('class_label', inplace=False, axis=1)
val_replaced = val_data.drop('class_label', inplace=False, axis=1)
    
class_dict = pd.Series.to_dict(train_data.class_label.value_counts())

# plt.figure(1)
# plt.subplot(2,1,1)
# plot=train_data.class_label.value_counts().plot.bar(ylabel="Instance Count", xlabel="Class", ylim=(0,220))

# for bar in plot.patches:
#     plot.annotate(format(bar.get_height(), '.2f'),
#                    (bar.get_x() + bar.get_width() / 2,
#                     bar.get_height()), ha='center', va='center',
#                    size=10, xytext=(0, 8),
#                    textcoords='offset points')
    
   
# plt.subplot(2,1,2)

# plot2=val_data.class_label.value_counts().plot.bar(ylabel="Instance Count", xlabel="Class",ylim=(0,79))

# for bar in plot2.patches:
#     plot2.annotate(format(bar.get_height(), '.2f'),
#                    (bar.get_x() + bar.get_width() / 2,
#                     bar.get_height()), ha='center', va='center',
#                    size=10, xytext=(0, 8),
#                    textcoords='offset points')
    

article_length = len(train_data)
pi = {}
T_0 = {}
    
for x, values in class_dict.items():
    pi[x] = values/article_length

for classes in (0,1,2,3,4):
    class_data = train_data[train_data.class_label==classes]
    class_data = class_data.drop('class_label', inplace=False, axis=1)
    amount_list = []
    for column in class_data:      
        amount_list.append(sum(class_data[column])) 
    T_0[classes] = amount_list

T_val = pd.DataFrame.from_dict(T_0, orient='index')
Theta_value = {}
Series_sum = T_val.sum(axis=1)
Series_sum_list = pd.Series.to_list(Series_sum)
Theta_value = T_val.div(Series_sum, axis=0)

class_prediction = {}
for file in range(len(val_replaced)):   
    prediction_values = []
    for clas in (0,1,2,3,4):
        a = np.nan_to_num(np.log(Theta_value.iloc[[clas]].values))
        y_value = np.log(pi[clas]) + np.sum(a*val_replaced.iloc[[file]].values)
        prediction_values.append(y_value)
    class_prediction[file] = prediction_values

predicted_classes = {}
for x in class_prediction:   
    max_index = class_prediction[x].index(max(class_prediction[x]))
    predicted_classes[x] = max_index

true_classes = val_data.pop('class_label')
t = 0
f = 0 
for x in predicted_classes:      
    if predicted_classes[x] == true_classes.values[x]:
        t += 1
    else:
        f += 1
    
print("Accuracy Percentage:",t/(t+f)*100)

conf_vec=np.zeros((5,5))
for x in range(len(val_replaced)):
    conf_vec[true_classes[x]][predicted_classes[x]] +=1
print("Confusion Vector:\n",conf_vec)

