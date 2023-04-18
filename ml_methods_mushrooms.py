#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 00:30:29 2023

@author: chrisfeng
"""
#General imports
import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np

#ML Models
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree

#Metrics and testing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

mushrooms = pd.read_csv('/Users/chrisfeng/Desktop/CS470_Final_Project/MushroomAnalysis/secondary_data_shuffled.csv', sep=';')

#Replace NAs
for col in mushrooms: 
    if(is_numeric_dtype(mushrooms[col])):
        mushrooms[col].fillna(mushrooms[col].mean(), inplace=True)
    else:
        mushrooms[col].fillna(mushrooms[col].mode()[0], inplace=True)

#Get y labeling
y = mushrooms['class']
y = y.apply(lambda x: x=="p").astype(int)
mushrooms.drop(columns=['class'], inplace=True)


#One hot encoding
mushrooms = pd.get_dummies(mushrooms)

#Split
x_train, x_test, y_train, y_test = train_test_split(mushrooms, y, test_size=0.3, random_state=42)
y_train = np.array(y_train)
y_test = np.array(y_test)

acc = {}
f1 = {}

logit = LogisticRegression(penalty='l2', solver='saga')
logit.fit(x_train, y_train)
yHat = logit.predict(x_test)
acc["logit"] = accuracy_score(y_test, yHat)
f1["logit"] = f1_score(y_test, yHat)


clf = Perceptron(tol=1e-3, random_state=0)
clf.fit(x_train, y_train)
yHat = clf.predict(x_test)
acc["perceptron"] = accuracy_score(y_test, yHat)
f1["perceptron"] = f1_score(y_test, yHat)

mlp = MLPClassifier(solver='lbfgs', alpha=1e-5,\
                    hidden_layer_sizes=(5, 2), random_state=1)
mlp.fit(x_train, y_train)
yHat = mlp.predict(x_test)
acc["nn"] = accuracy_score(y_test, yHat)
f1["nn"] = f1_score(y_test, yHat)

nb = BernoulliNB()
nb.fit(x_train,y_train)
yHat = nb.predict(x_test)
acc["nb"] = accuracy_score(y_test, yHat)
f1["nb"] = f1_score(y_test, yHat)

dt = DecisionTreeClassifier(min_samples_leaf=20)
dt.fit(x_train, y_train)
yHat = dt.predict(x_test)
acc["dt"] = accuracy_score(y_test, yHat)
f1["dt"] = f1_score(y_test, yHat)

print(acc)
print(f1)



