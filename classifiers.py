import sys
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

def getAccuracy(first,second):
    same=np.equal(first,second)
    top=np.sum(same)
    print(top)
    return (top/np.shape(first)[0])

X_train = np.load('madelon/train-X.npy')
y_train = np.load('madelon/train-y.npy')
X_test = np.load('madelon/train-X.npy')
y_test = np.load('madelon/train-y.npy')

tumors=np.shape(X_train)[0]
genes=np.shape(X_train)[1]

SGD_model = SGDClassifier(loss='log', max_iter=10000)
model = SGD_model.fit(X_train, y_train)
y_prime = model.predict(X_test)
print(getAccuracy(y_prime,y_test))

tree = DecisionTreeClassifier(criterion='entropy')
model = tree.fit(X_train, y_train)
y_prime = model.predict(X_test)
print(getAccuracy(y_prime,y_test))

stump = DecisionTreeClassifier(criterion='entropy',max_depth=4)
model = stump.fit(X_train, y_train)
y_prime = model.predict(X_test)
print(getAccuracy(y_prime,y_test))

stumps=[]
features=[]
stump_tree=np.zeros((tumors,50))
for j in range(50):
    half=int(genes/2)
    my_features=np.random.choice(genes,size=half,replace=False)
    features.append(my_features)
    subsetted_X_train=X_train[:,my_features]
    stump = DecisionTreeClassifier(criterion='entropy',max_depth=4).fit(subsetted_X_train, y_train)
    stumps.append(stump)
    stump_output = stump.predict(subsetted_X_train)
    stump_tree[:,j] = stump_output

def transformMyData(X,stumps,tumors,features):
    output=np.zeros((tumors,50))
    for j in range(50):
        j_features=features[j]
        X_subset=X[:,j_features]
        output[:,j] = stumps[j].predict(X_subset)
    return(output)

model4 = SGDClassifier(loss='log', max_iter=10000).fit(stump_tree,y_train)
X_prime_test=transformMyData(X_test,stumps,tumors,features)
y_prime = model4.predict(X_prime_test)
print(getAccuracy(y_prime,y_test))
