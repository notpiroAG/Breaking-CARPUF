import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.linear_model import RidgeClassifier
import time as tm
from scipy.linalg import khatri_rao

# You are allowed to import any submodules of sklearn that learn linear models e.g. sklearn.svm etc
# You are not allowed to use other libraries such as keras, tensorflow etc
# You are not allowed to use any scipy routine other than khatri_rao

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py

# DO NOT CHANGE THE NAME OF THE METHODS my_fit, my_map etc BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length

#choosing either LinearSVC or LogisticRegression or RidgeClassifier

# clf = LinearSVC(C=1.0, max_iter=10000, tol=1e-3, dual=False)
# clf = RidgeClassifier(alpha=1.0, max_iter=10000, tol=1e-5)
clf = LogisticRegression(C=1.0, max_iter=10000, tol=1e-5)


################################
# find_feat function takes a row of data and returns the feature vector for that row
def find_feat(row):
	temp = []
	prev = 1
	for c_i in row:
		prev *= (1 - 2 * c_i)
		temp.append(prev)
	temp = temp[::-1]

	row_feat = []
	for i  in range(len(temp)):
		for j in range(i+1, len(temp)):
			row_feat.append(temp[i] * temp[j])
		
	for i in range(len(temp)):
		row_feat.append(temp[i])

	return row_feat
################################


################################
# Non Editable Region Starting #
################################
def my_fit( X_train, y_train ):
################################
#  Non Editable Region Ending  #
################################

	X_train = my_map(X_train)

	#training the model here 
	clf.fit(X_train, y_train)

	w = clf.coef_.flatten()
	b = clf.intercept_

	return w, b
################################


################################
# Non Editable Region Starting #
################################
def my_map( X ):
################################
#  Non Editable Region Ending  #
################################

	feat = []
	for row in X:
		row_feat = find_feat(row)
		feat.append(row_feat)
	return np.array(feat)
################################


#loading data from the files
train_data = np.loadtxt('train.dat')
test_data = np.loadtxt('test.dat')

#stroing the features and labels in X_train, y_train
X_train, y_train = train_data[:, :32], train_data[:, 32]
X_test, y_test = test_data[:, :32], test_data[:, 32]



#training model
tic = tm.perf_counter()
w, b = my_fit(X_train, y_train)
tok = tm.perf_counter()

print("Time taken to train:", tok - tic)

#mapping test data
tic = tm.perf_counter()
feat = my_map(X_test)
tok = tm.perf_counter()

print("Time taken to map:", tok - tic)

#predicting labels of test data using the model
scores = feat.dot( w ) + b
pred = np.zeros_like( scores )
pred[scores > 0] = 1

#checking accuracy of the model with test labels
accuracy  = 0
accuracy += np.average( y_test == pred )
print("Accuracy:", accuracy)