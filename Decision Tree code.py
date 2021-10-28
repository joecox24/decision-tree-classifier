# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 11:31:14 2020

@author: joewc
"""
import numpy as np
from numpy import inf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn import tree
from collections import Counter
from itertools import combinations
import time
import math 
from sklearn.datasets import make_classification


class DecisionTree :
    # Initiates the DecisionTree class, class of each data point should be the
    # final variable in the list.
   def __init__(self, depth, data, datatype = [0]): 
       self.depth = depth
       self.data = np.array(data)
       self.datatype = datatype
       self.content = None
       self.left = None
       self.right = None
       # Creates a list of the different classes in the dataset.
       self.classes = list(set([x[-1] for x in data]))
       
       # Increases the length of the datatype list in the case that the list 
       # given by the user is too short. 
       if len(datatype) < len(self.data[0]) - 1 :
           self.datatype = datatype*(len(self.data[0]) - 1)
           
   # Calculates the gini impurity for one half of a partition.       
   def Gini(self, data):  
    
    labels = [x[-1] for x in data]
    n = len(data)
    gini = 1
    if n == 0:
        return 1
    
    # Counts the number of data points belonging to each class.
    class_counts = [0]*len(self.classes)
    for i in range(len(labels)) :
        for j in range(len(self.classes)) :
            if labels[i] == self.classes[j] :
                class_counts[j] += 1
                break
    
    for k in range(len(self.classes)):
        gini = gini - (class_counts[k]/n)**2
    return gini

 
   # Calculates the Gini impurity of a partition using the Gini impurity for 
   # each half of the partition.
   def combinedGini(self, left, right):
    
      totalLength = len(left) + len(right)
      return (len(left)/totalLength) * self.Gini(left) + (len(right)/totalLength) * self.Gini(right)


   # Calculates the entropy for one half of the partition.
   def Entropy(self,data):
    
    labels = [x[-1] for x in data]
    n = len(data)
    entropy = 0
    if n == 0:
        return 1
        
    class_counts = [0]*len(self.classes)
    for i in range(len(labels)) :
        for j in range(len(self.classes)) :
            if labels[i] == self.classes[j] :
                class_counts[j] += 1
                break
            
    # Iteratively calculates the entropy by summing the term for each class.       
    for k in range(len(self.classes)):
        p_class = class_counts[k]/n
        if p_class == 0:
            continue
        else:
          entropy = entropy - p_class*math.log(p_class,2)       
    
    return entropy  


   # Calculates the entropy for a partition using the entropy for each half.
   def combinedEntropy(self, left, right):
       totalLength = len(left) + len(right)
       return (len(left)/totalLength) * self.Entropy(left) + (len(right)/totalLength) * self.Entropy(right)
   
   # Calculates the negative of the chi-squared test statistic.
   def ChiSquared(self, data):  
    
    labels = [x[-1] for x in data]
    n = len(data)
    chiSquared = 0
    if n == 0:
        return 1
    
    data_class_counts = [0]*len(self.classes)
    for i in range(len(labels)) :
        for j in range(len(self.classes)) :
            if labels[i] == self.classes[j] :
                data_class_counts[j] += 1
                break
    
    # Iteratively calculates the chi-squared test statistic.        
    for k in range(len(self.classes)):
      expected = (self.class_counts[self.classes[k]]/len(self.data))*n
      chiSquared += math.sqrt((data_class_counts[k] - expected)**2/expected)
     
    # Returns the negative of the test statistic.
    return -chiSquared  
   
    
   # Sums the chi-squared test statistic for each half of a partition
   def combinedChiSquared(self, left, right):
       return self.ChiSquared(left) + self.ChiSquared(right)
   
    
   # Finds the best data partition based on the user defined splitting criterion
   def _findBestSplit(self, criterion):
       if criterion == 'Entropy':
          cri = self.combinedEntropy
       
       elif criterion == 'ChiSquared':
           cri = self.combinedChiSquared
           
           # Calculates the number of data points of each class in the dataset 
           # being split as it is needed to calculate the test statistic.
           self.class_counts = Counter([x[-1] for x in self.data])
       else:
           cri = self.combinedGini
          
       
       score = inf 
       
       data = self.data
       n = len(self.data[0]) - 1
       
       left = None  
       right = data  
        
       split = - inf
       feature = -1
       
       # The following for loop cycles through all of the features in the dataset
       for j in range(n):
         
         # Case when the feature is numeric
         if self.datatype[j] == 0:  
           
            orderedData = sorted(data, key = lambda x: x[j])  
           
            # Tests all possible binary partitions of the data
            for i in range(1,len(self.data)-1):  
             tempLeft = orderedData[0:i]   # Splits the data into a binary partition 
             tempRight = orderedData[i::]
             
             # Checks that data points with the same value are in the same half of the partition
             if (tempLeft[-1][j] != tempRight[0][j]):
               tempScore = cri(tempLeft, tempRight)  
           
                # Checks whether the Gini index of a partition is smaller than 
                # the current Gini index
               if (tempScore < score):  
                  score = tempScore # Reassigns the current smallest Gini index
                  left = tempLeft # Reassigns the left half of the partition
                  right = tempRight  # Reassigns the right half of the partition
                  split = (float(left[-1][j]) + float(right[0][j]))/2
                  feature = j
               
  
         # Case when the feature is categorical   
         elif self.datatype[j] == 1:
             pos_values = set(self.data[:,j])
             for k in range(1, len(pos_values) - 1):
                # Finds all combinations of the possible values of the 
                # categorical variables
                combs = list(combinations(pos_values, k))
                
                # Cycles through the combinations of the possible values
                for comb in combs:
                    tempLeft = self.data[[x[j] in comb for x in self.data]]
                    tempRight = self.data[[x[j] not in comb for x in self.data]]
                    
                    # Calculates the score for the appropriate criterion
                    tempScore = cri(tempLeft,tempRight)
                    
                    # Updates the output variables if the new score is the lowest
                    if (tempScore < score):
                        score = tempScore
                        left = tempLeft
                        right = tempRight
                        
                        # split variable is assigned to the combination of 
                        # variables in the left half of the partition
                        split = comb   
                        feature = j
                    
                           
       if split != -inf:   
             return (score, split, feature, left, right) 
       
        # Returns false if a split cannot be found
       return False
   
   # Trains the tree using the chosen splitting criterion
   def fit(self, criterion = 'Gini'):
       
       # Calculates the best split of the data
       best_split = self._findBestSplit(criterion)
       if self.depth > 0 and best_split != False:
          
          # Stores the split point and the feature being tested at the node
          self.content = [best_split[1], best_split[2]]
          
          # Left and Right branches are trained recursively with the depth reduced by 1
          self.left = DecisionTree(self.depth - 1, best_split[3],
                                   datatype = self.datatype)
          self.right = DecisionTree (self.depth - 1, best_split[4],
                                     datatype = self.datatype) 
          
          self.left.fit(criterion)
          self.right.fit(criterion)
       
       else:
          # When the depth reaches zero the most common label at each node is stored
          labels = [x[-1] for x in self.data]
          self.content = Counter(labels).most_common(1)[0][0]
   
   # Classifies a single data point        
   def classify(self, data):
         if self.right != None and self.left != None:
          
           # Case when tested feature is numeric  
           if self.datatype[self.content[1]] == 0:
             # Tests whether the value of the feature is less than the split value
             if float(data[self.content[1]]) < self.content[0]:
               return self.left.classify(data)
             
             else:
                return self.right.classify(data)
           
            
           # Case when tested feature is categorical
           elif  self.datatype[self.content[1]] == 1:
               # Tests whether the feature is in the combination of values in 
               # the left hand of the split
               if data[self.content[1]] in self.content[0]:
                  return self.left.classify(data)
               else:
                  return self.right.classify(data)
         else: 
             return self.content
   

   # Predicts the label of multiple data points by calling the classify method
   # and returns a list of the predicted labels for each data point.
   def predict(self,data):
       n = len(data)
       labels = [None] * n
       for i in range(n):
           labels[i] = self.classify(data[i])
       return labels    
 

## Test Cases ##

# Generates a dataset of size 10 with 2 features and 2 possible labels for each data point           
test_case, test_case_labels = make_classification(n_samples = 10, n_features = 2,
                                                  n_redundant = 0)

# Attaches the labels to the data for training the decision tree.
train_data = np.column_stack((np.array(test_case), np.array(test_case_labels)))


# A tree of depth 1 is trained and the labels are predicted for each splitting criterion
dt = DecisionTree(depth = 1, data = train_data)
dt.fit('Gini')
dt.predict(test_case)

dt.fit('Entropy')
dt.predict(test_case)

dt.fit('ChiSquared')
dt.predict(test_case)


# Generates a dataset of size 10 with 3 features and 3 possible labels for each data point
test_case, test_case_labels = make_classification(n_samples = 10, n_informative = 3, 
                                                  n_redundant = 0, n_features = 3,
                                                  n_classes = 3)

train_data = np.column_stack((np.array(test_case), np.array(test_case_labels)))

dt = DecisionTree(depth = 4, data = train_data)
dt.fit('Gini')
dt.predict(test_case)

dt.fit('Entropy')
dt.predict(test_case)

dt.fit('ChiSquared')
dt.predict(test_case)


## Decision tree analysis ##

## WARNING: the following code will take a long time to run due to repeatedly 
## training a large number of decision trees with different datasets


df = pd.read_csv("data_banknote_authentication.csv", header = None)   
banknote_data = df.values.tolist()

Time = []
Accuracy = []
Criterion = []   
Depth = []

train_sizes = [ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
for size in train_sizes:
  train_data, test_data = train_test_split(banknote_data, train_size = size)
  train_labels = [x[-1] for x in train_data]
  test_labels = [x[-1] for x in test_data]
  test_data = pd.DataFrame(test_data)[[0,1,2,3]].values.tolist()
  for i in range(1,8):
   for crit in ['Gini', 'Entropy', 'ChiSquared']:  
     start_time = time.time()
     dt = DecisionTree(i,train_data)
     dt.fit(criterion = crit)
     end_time = time.time()
     Time.append(end_time - start_time)
     Depth.append(i)
     predicted_labels = dt.predict(test_data)
     Accuracy.append(accuracy_score(y_true = test_labels, y_pred = predicted_labels))
     Criterion.append(crit)
   
   start_time = time.time()
   skdt = tree.DecisionTreeClassifier(max_depth = i, criterion = 'gini')
   skdt.fit(pd.DataFrame(train_data)[[0,1,2,3]].values.tolist(), train_labels)
   end_time = time.time()

   Time.append(end_time - start_time) 
   Depth.append(i)
   Accuracy.append(accuracy_score(y_true = test_labels, y_pred = skdt.predict(test_data)))
   Criterion.append('sklearn_gini')   

   start_time = time.time()
   skdt = tree.DecisionTreeClassifier(max_depth = i, criterion = 'entropy')
   skdt.fit(pd.DataFrame(train_data)[[0,1,2,3]].values.tolist(), train_labels)
   end_time = time.time()

   Time.append(end_time - start_time) 
   Depth.append(i)
   Accuracy.append(accuracy_score(y_true = test_labels, y_pred = skdt.predict(test_data)))
   Criterion.append('sklearn_entropy')  

training_props = [0.1]*35 + [0.2]*35 + [0.3]*35 + [0.4]*35 + [0.5]*35 + [0.6]*35 + [0.7]*35 + [0.8]*35 + [0.9]*35
banknote_test = pd.DataFrame({'Criterion' : Criterion, 'Training Proportion' : training_props,
              'Tree Depth' : Depth, 'Accuracy' : Accuracy, 'Training Time' : Time}) 

banknote_test.to_csv('banknote_test.csv') 
 

df = pd.read_csv("bezdekiris.csv", header = None)   
IRIS_data = df.values.tolist()

Time = []
Accuracy = []
Criterion = []   
Depth = []
Precision_1 = []
Precision_2 = []
Precision_3 = []
Recall_1 = []
Recall_2 = []
Recall_3 = []
F1score_1 = []
F1score_2 = []
F1score_3 = []

for k in range(10):
  train_data, test_data = train_test_split(IRIS_data, train_size = 0.8)
  train_labels = [x[-1] for x in train_data]
  test_labels = [x[-1] for x in test_data]
  test_data = pd.DataFrame(test_data)[[0,1,2,3]].values.tolist()
  for i in range(1,8):
   for crit in ['Gini', 'Entropy', 'ChiSquared']:  
     start_time = time.time()
     dt = DecisionTree(i,train_data)
     dt.fit(criterion = crit)
     end_time = time.time()
     Time.append(end_time - start_time)
     Depth.append(i)
     predicted_labels = dt.predict(test_data)
     Accuracy.append(accuracy_score(y_true = test_labels, y_pred = predicted_labels))
     
     F1 = f1_score(y_true = test_labels, y_pred = predicted_labels, average = None)
     precision = precision_score(y_true = test_labels, y_pred = predicted_labels, average = None)
     recall = recall_score(y_true = test_labels, y_pred = predicted_labels, average = None)
     F1score_1.append(F1[0])
     F1score_2.append(F1[1])
     F1score_3.append(F1[2])
     Precision_1.append(precision[0])
     Precision_2.append(precision[1])
     Precision_3.append(precision[2])
     Recall_1.append(recall[0])
     Recall_2.append(recall[1])
     Recall_3.append(recall[2])
     
     Criterion.append(crit)
     
   
   start_time = time.time()
   skdt = tree.DecisionTreeClassifier(max_depth = i, criterion = 'gini')
   skdt.fit(pd.DataFrame(train_data)[[0,1,2,3]].values.tolist(), train_labels)
   end_time = time.time()

   Time.append(end_time - start_time) 
   Depth.append(i)
   predicted_labels = skdt.predict(test_data)
   Accuracy.append(accuracy_score(y_true = test_labels, y_pred = predicted_labels ))
   F1 = f1_score(y_true = test_labels, y_pred = predicted_labels, average = None)
   precision = precision_score(y_true = test_labels, y_pred = predicted_labels, average = None)
   recall = recall_score(y_true = test_labels, y_pred = predicted_labels, average = None)
   F1score_1.append(F1[0])
   F1score_2.append(F1[1])
   F1score_3.append(F1[2])
   Precision_1.append(precision[0])
   Precision_2.append(precision[1])
   Precision_3.append(precision[2])
   Recall_1.append(recall[0])
   Recall_2.append(recall[1])
   Recall_3.append(recall[2])
   Criterion.append('sklearn_gini')   

   start_time = time.time()
   skdt = tree.DecisionTreeClassifier(max_depth = i, criterion = 'entropy')
   skdt.fit(pd.DataFrame(train_data)[[0,1,2,3]].values.tolist(), train_labels)
   end_time = time.time()

   Time.append(end_time - start_time) 
   Depth.append(i)
   predicted_labels = skdt.predict(test_data)
   Accuracy.append(accuracy_score(y_true = test_labels, y_pred = predicted_labels))
   F1 = f1_score(y_true = test_labels, y_pred = predicted_labels, average = None)
   precision = precision_score(y_true = test_labels, y_pred = predicted_labels, average = None)
   recall = recall_score(y_true = test_labels, y_pred = predicted_labels, average = None)
   F1score_1.append(F1[0])
   F1score_2.append(F1[1])
   F1score_3.append(F1[2])
   Precision_1.append(precision[0])
   Precision_2.append(precision[1])
   Precision_3.append(precision[2])
   Recall_1.append(recall[0])
   Recall_2.append(recall[1])
   Recall_3.append(recall[2])
   Criterion.append('sklearn_entropy')

Iris_test = pd.DataFrame({'Criterion' : Criterion, 'Tree Depth' : Depth, 
                          'Training Time' : Time, 'Accuracy' : Accuracy,
                          'Setosa F1' : F1score_1, 'Versicolor F1' : F1score_2,
                          'Virginica F1' : F1score_3, 'Setosa Precision' : Precision_1,
                          'Versicolor Precision' : Precision_2, 'Virginica Precision' : Precision_3,
                          'Setosa Recall' : Recall_1, 'Versicolor Recall' : Recall_2,
                          'Virginica Recall' : Recall_3})

Iris_test.to_csv('iris_test.csv')




df = pd.read_csv("crx.csv", header = None)
for column in df.columns:
    df = df[df[column] != '?']

credit_data = df.values.tolist()
credit_datatypes = [1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0]

Time = []
Accuracy = []
Criterion = []   
Depth = []
F1_positive = []
F1_negative = []

for j in range(5):
  train_data, test_data = train_test_split(credit_data, train_size = 0.1)
  train_labels = [x[-1] for x in train_data]
  test_labels = [x[-1] for x in test_data]
  test_data = pd.DataFrame(test_data)[df.columns[:-1]].values.tolist()
  for i in range(1,6):
   for crit in ['Gini', 'Entropy', 'ChiSquared']:  
     start_time = time.time()
     dt = DecisionTree(i,train_data, credit_datatypes)
     dt.fit(criterion = crit)
     end_time = time.time()
     Time.append(end_time - start_time)
     Depth.append(i)
     predicted_labels = dt.predict(test_data)
     Accuracy.append(accuracy_score(y_true = test_labels, y_pred = predicted_labels))
     
     F1 = f1_score(y_true = test_labels, y_pred = predicted_labels, average = None)
     F1_positive.append(F1[0])
     F1_negative.append(F1[1])
             
     Criterion.append(crit)

Credit_test = pd.DataFrame({'Criterion' : Criterion, 'Tree Depth' : Depth, 
                          'Training Time' : Time, 'Accuracy' : Accuracy,
                          'Positive F1' : F1_positive, 'Negative F1' : F1_negative})

Credit_test.to_csv('credit_test.csv')
     
