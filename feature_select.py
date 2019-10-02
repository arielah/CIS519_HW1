import sys
import numpy as np
import pandas as pd
from string import ascii_lowercase
from cross_validation import getAccuracy
from cross_validation import train_and_evaluate_sgd 
from cross_validation import train_and_evaluate_decision_tree
from cross_validation import train_and_evaluate_decision_stump
from cross_validation import train_and_evaluate_sgd_with_stumps

train_names=pd.read_csv("badges/train.names.txt",header=None)
y_train=np.load("badges/train.labels.npy")
test_names=pd.read_csv("badges/test.names.txt",header=None)
y_test=np.load("badges/test.labels.npy")

def compute_features(names):
	LETTERS = {letter: index for index, letter in enumerate(ascii_lowercase, start=0)} 

	features=np.zeros((names.shape[0],260))
	for i in range(names.shape[0]):
		name=names.ix[i,0]
		name = name.lower()
		firstname,lastname=name.split(' ')
		if len(firstname)>5:
			firstname=firstname[0:5]
		numbers = np.array([LETTERS[character] for character in firstname])
		offset=np.arange(0,len(numbers)*26,26)
		numbers=numbers+offset
		features[i,numbers]=1
		if len(lastname)>5:
			lastname=lastname[0:5]
		numbers = np.array([LETTERS[character] for character in lastname])
		offset=np.arange(5*26,(5+len(numbers))*26,26)
		numbers=numbers+offset
		features[i,numbers]=1
	return(features)

X_train=compute_features(train_names)
X_test=compute_features(test_names)

big_ugh=train_and_evaluate_sgd(X_train, y_train, X_test, y_test)
print(big_ugh)
big_yuck=train_and_evaluate_decision_tree(X_train, y_train, X_test, y_test)
print(big_yuck)
big_gross=train_and_evaluate_decision_stump(X_train, y_train, X_test, y_test)
print(big_gross)
big_gah=train_and_evaluate_sgd_with_stumps(X_train, y_train, X_test, y_test)
print(big_gah)
