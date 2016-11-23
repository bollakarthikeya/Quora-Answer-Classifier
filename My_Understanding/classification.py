
import re
import pandas
import numpy
from sklearn import metrics
from sklearn import preprocessing
from sklearn.svm import LinearSVC 
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import KFold
from sklearn.metrics import classification_report

# reading input
fp = open("input00.txt", "r")
count = int(0)
train = []
test = []
for i in fp:
	if ":" in i and count is int(1):	# we are reading training data
		train.append(i)	
	elif ":" in i and count is int(2):	# we are reading test data
		test.append(i)
	else:
		count = count + int(1)

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
train_df = train_df.drop(train_df.columns[[0,4,11,14,15,16,17,18,20,23,24]], axis = 1)	# dropping specific columns

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
test_df = test_df.drop(test_df.columns[[0,3,10,13,14,15,16,17,19,22,23]], axis = 1)	# dropping specific columns

# ===============================================
# TEST DATA ACTUAL RESULTS
# ===============================================
fp = open("output00.txt", "r")
test_results = []
for i in fp:
	temp = i.split(" ")
	test_results.append(re.sub("\n", "", temp[1]))
fp.close()
test_results_df = pandas.DataFrame(test_results)

# ===============================================
# TRAINING THE MODEL
# ===============================================
X = train_df[[2,3,5,6,7,8,9,10,12,13,19,21,22]]
y = train_df[[1]]

# standardize
X = preprocessing.scale(X)

# defining the classification model
sv = LinearSVC()

# classification using cross-validation (10-fold)
kf = KFold(len(X), n_folds = 10)

for train_index, test_index in kf:
	X_train, X_test = X[train_index[0] : train_index[-1]], X[test_index[0] : test_index[-1]]
	y_train, y_test = y[train_index[0] : train_index[-1]], y[test_index[0] : test_index[-1]]
	sv.fit(X_train, numpy.ravel(y_train))
	y_pred = sv.predict(X_test)
	
	"""
	report = classification_report(y_test, y_pred)
	temp = report.split()
	accuracy = accuracy_score(y_test, y_pred)
	print "=============================================="
	print "Accuracy: ",accuracy
	print "Precision: ", float(temp[17])
	print "Recall: ", float(temp[18])
	print "F-Score: ", float(temp[19])	# write the f1-score during each iteration
	print "=============================================="
	"""

# ===============================================
# TESTING THE MODEL
# ===============================================
test_df = preprocessing.scale(test_df)
test_pred_df = sv.predict(test_df)

report = classification_report(test_results_df, test_pred_df)
print report
temp = report.split()
accuracy = accuracy_score(test_results_df, test_pred_df)
print "Accuracy: ",accuracy
print "Precision: ", float(temp[17])
print "Recall: ", float(temp[18])
print "F-Score: ", float(temp[19])	# write the f1-score during each iteration









