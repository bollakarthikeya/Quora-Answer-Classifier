
# KARTHIKEYA BOLLA
# QUORA ANSWER CLASSIFIER
# Classification done using Support Vector Machines (Linear Kernel)

import re
import pandas
import numpy
import sys
from sklearn import metrics
from sklearn import preprocessing
from sklearn.svm import LinearSVC 
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import KFold
from sklearn.metrics import classification_report

# ===============================================
# READING INPUT FROM STDIN
# ===============================================
train = []
test = []

# reading the line '4500 23'
inp = raw_input()
N, M = inp.split(" ")
N = int(N)
M = int(M)
counter = int(0)
while counter < N:
	i = raw_input()
	train.append(i)	
	counter = counter + 1

# reading the line '500'
inp = int(raw_input())
counter = int(0)
while counter < inp:
	i = raw_input()
	test.append(i)	
	counter = counter + 1

# ===============================================
# PROCESSING TRAINING DATA
# ===============================================
for i in range(len(train)):
	temp = train[i].split(" ")
	# change all 23 features
	for j in range(2,len(temp)):
		temp[j] = temp[j].split(":")[1]
	# merge all 
	train[i] = temp
train_df = pandas.DataFrame(train)	# convert to data frame
train_df = train_df.drop(train_df.columns[[0,4,11,14,15,16,17,18,20,23,24]], axis = 1)	# dropping specific columns as they are binary attributes

# ===============================================
# PROCESSING TEST DATA
# ===============================================
for i in range(len(test)):
	temp = test[i].split(" ")
	# change all 23 features
	for j in range(1,len(temp)):
		temp[j] = temp[j].split(":")[1]
	# merge all
	test[i] = temp
test_df = pandas.DataFrame(test)	# convert to data frame
test_df_ans_id = list(test_df[0])	# storing the answer ids
test_df = test_df.drop(test_df.columns[[0,3,10,13,14,15,16,17,19,22,23]], axis = 1)	# dropping specific columns


# ===============================================
# TRAINING THE MODEL
# ===============================================
X = train_df[[2,3,5,6,7,8,9,10,12,13,19,21,22]]
y = train_df[[1]]

# standardize
X = preprocessing.scale(X)

# defining the classification model
sv = LinearSVC()

# fitting the model
sv.fit(X, y)

# ===============================================
# TESTING THE MODEL
# ===============================================
test_df = preprocessing.scale(test_df)
test_pred_df = sv.predict(test_df)
test_pred_df_list = list(test_pred_df)
for i in range(len(test_pred_df_list)):
	 sys.stdout.write(test_df_ans_id[i] + " " + test_pred_df_list[i] + "\n")	# writing using STDOUT




