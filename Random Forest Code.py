import csv
import random
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.datasets import make_friedman1
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

# names of our training testing splits of our data
dataset = 'hptrain.csv'
dataset2 = 'hptest.csv'
csvreader = csv.reader(open(dataset))
header = next(csvreader)[1:]
# split the labels and attributes or training and testing
trainingX = []
trainingY = []
for line in csvreader:
    trainingX.append(line[1:])
    trainingY.append(line[0])

csvreader = csv.reader(open(dataset2))
header2 = next(csvreader)[1:]
testX = []
testY = []
for line in csvreader:
    testX.append(line[1:])
    testY.append(line[0])
# create the random forest model and train it
clf = RandomForestClassifier()
clf.fit(trainingX, trainingY)
# determine accuracy for basic random forest model
predictValues = clf.predict(testX)
count = 0
for i in range(len(predictValues)):
    print(predictValues[i], testY[i])
    if predictValues[i] == testY[i]:
        count += 1
print(count / len(predictValues))
# output classification report
target_names = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
                "November", "December"]
print(classification_report(testY, predictValues, target_names=target_names))

featToRank = {}
# create random forest classifier and determine attribute imporantance with RFE
estimator = RandomForestClassifier()
selector = RFE(estimator, n_features_to_select=1, step=1)
selector = selector.fit(trainingX, trainingY)
for i in range(len(header)):
    featToRank[header[i]] = selector.ranking_[i]
# run through testing set with new decision tree voting scheme
finalans = []
totalct = 0
myTrees = []

#create trees
for j in range(100):
    tree = DecisionTreeClassifier().fit(trainingX, trainingY)
    myTrees.append(tree)
for k in range(len(testX)):
    predictTotal = {}
    for j in range(100):
        #obtain decision path
        decision_path = myTrees[j].decision_path([testX[k]])
        p = myTrees[j].tree_
        level_length = len(decision_path.indices)
        i = 1
        #find features
        featuresfound = set()
        for node_id in decision_path.indices:
            # Ignore last level because it is the last node
            # without decision criteria or rule
            if i < level_length:
                col_name = header[p.feature[node_id]]
                threshold_value = p.threshold[node_id]
                featuresfound.add(col_name)
            i = i + 1
        count = 0
        nume = 0
        denom = 0
        for val in featuresfound:
            count += 1
            nume += count
            denom += featToRank[val]
        answ = (int)(myTrees[j].predict([testX[k]]))
        votingweight = 1 + (nume / denom)
        if answ not in predictTotal:
            predictTotal[answ] = votingweight
        else:
            predictTotal[answ] += votingweight
    maxi = 0
    pansw = 0
    # find correct prediction
    for val in predictTotal:
        if predictTotal[val] > maxi:
            maxi = predictTotal[val]
            pansw = val
    if pansw == int(testY[k]):
        totalct += 1
    finalans.append(str(pansw))
    print(pansw, testY[k])

# print classification report and accuracy
print(classification_report(testY, finalans, target_names=target_names))
print(totalct / 72)
