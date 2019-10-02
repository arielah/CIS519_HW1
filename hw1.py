import sys
import numpy as np
import pandas as pd
from string import ascii_lowercase

names=pd.read_csv("badges/test.names.txt",header=None)

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