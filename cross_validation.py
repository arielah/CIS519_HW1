import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

def getAccuracy(first,second):
    same=np.equal(first,second)
    top=np.sum(same)
    return (top/np.shape(first)[0])


def train_and_evaluate_sgd(X_train, y_train, X_test, y_test):
    SGD_model = SGDClassifier(loss='log', max_iter=10000)
    model = SGD_model.fit(X_train, y_train)
    y_prime_train = model.predict(X_train)
    part1=getAccuracy(y_prime_train, y_train)
    y_prime_test = model.predict(X_test)
    part2=getAccuracy(y_prime_test, y_test)
    return part1, part2

def train_and_evaluate_decision_tree(X_train, y_train, X_test, y_test):
    tree_model = DecisionTreeClassifier(criterion='entropy')
    model = tree_model.fit(X_train, y_train)
    y_prime_train = model.predict(X_train)
    part1=getAccuracy(y_prime_train, y_train)
    y_prime_test = model.predict(X_test)
    part2=getAccuracy(y_prime_test, y_test)
    return part1, part2

def train_and_evaluate_decision_stump(X_train, y_train, X_test, y_test):
    stump_model = DecisionTreeClassifier(criterion='entropy', max_depth=4)
    model = stump_model.fit(X_train, y_train)
    y_prime_train = model.predict(X_train)
    part1=getAccuracy(y_prime_train, y_train)
    y_prime_test = model.predict(X_test)
    part2=getAccuracy(y_prime_test, y_test)
    return part1, part2

def train_and_evaluate_sgd_with_stumps(X_train, y_train, X_test, y_test):
    ncols=np.shape(X_train)[1]
    stumps=[]
    features=[]
    stump_tree=np.zeros((np.shape(X_train)[0],50))
    for j in range(50):
        half=int(ncols/2)
        my_features=np.random.choice(ncols,size=half,replace=False)
        features.append(my_features)
        subsetted_X_train=X_train[:,my_features]
        stump = DecisionTreeClassifier(criterion='entropy',max_depth=4).fit(subsetted_X_train, y_train)
        stumps.append(stump)
        stump_output = stump.predict(subsetted_X_train)
        stump_tree[:,j] = stump_output
    output=np.zeros((np.shape(X_test)[0],50))
    for j in range(50):
        j_features=features[j]
        X_subset=X_test[:,j_features]
        output[:,j] = stumps[j].predict(X_subset)
    model4 = SGDClassifier(loss='log', max_iter=50000).fit(stump_tree,y_train)
    y_prime_train = model4.predict(stump_tree)
    part1=getAccuracy(y_prime_train, y_train)
    y_prime_test = model4.predict(output)
    part2=getAccuracy(y_prime_test, y_test)
    return part1, part2

def plot_results(sgd_train_acc, sgd_train_std, sgd_heldout_acc, sgd_heldout_std, sgd_test_acc,dt_train_acc, dt_train_std, dt_heldout_acc, dt_heldout_std, dt_test_acc,dt4_train_acc, dt4_train_std, dt4_heldout_acc, dt4_heldout_std, dt4_test_acc,stumps_train_acc, stumps_train_std, stumps_heldout_acc, stumps_heldout_std, stumps_test_acc):
    train_x_pos = [0, 4, 8, 12]
    cv_x_pos = [1, 5, 9, 13]
    test_x_pos = [2, 6, 10, 14]
    ticks = cv_x_pos

    labels = ['sgd', 'dt', 'dt4', 'stumps (4 x 50)']

    train_accs = [sgd_train_acc, dt_train_acc, dt4_train_acc, stumps_train_acc]
    print(train_accs)
    train_errors = [sgd_train_std, dt_train_std, dt4_train_std, stumps_train_std]

    cv_accs = [sgd_heldout_acc, dt_heldout_acc, dt4_heldout_acc, stumps_heldout_acc]
    cv_errors = [sgd_heldout_std, dt_heldout_std, dt4_heldout_std, stumps_heldout_std]

    test_accs = [sgd_test_acc, dt_test_acc, dt4_test_acc, stumps_test_acc]

    fig, ax = plt.subplots()
    ax.bar(train_x_pos, train_accs, yerr=train_errors, align='center', alpha=0.5, ecolor='black', capsize=10, label='train')
    ax.bar(cv_x_pos, cv_accs, yerr=cv_errors, align='center', alpha=0.5, ecolor='black', capsize=10, label='held-out')
    ax.bar(test_x_pos, test_accs, align='center', alpha=0.5, capsize=10, label='test')
    ax.set_ylabel('Accuracy')
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_title('Models')
    ax.yaxis.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.savefig("graph.png")


#Cross-validation

model1_train=np.zeros(5)
model1_heldout=np.zeros(5)
model2_train=np.zeros(5)
model2_heldout=np.zeros(5)
model3_train=np.zeros(5)
model3_heldout=np.zeros(5)
model4_train=np.zeros(5)
model4_heldout=np.zeros(5)

for holdout in range(5):
    x_train = None
    y_train = None
    for i in range(5):
        if i==holdout:
            x_test = np.load('madelon/cv-heldout-X.'+str(i)+'.npy')
            y_test = np.load('madelon/cv-heldout-y.'+str(i)+'.npy')
        else:
            x_part = np.load('madelon/cv-train-X.'+str(i)+'.npy')
            if x_train is not None:
                x_train = np.vstack((x_train,x_part))
            else:
                x_train = x_part
            y_part = np.load('madelon/cv-train-y.'+str(i)+'.npy')
            if y_train is not None:
                y_train = np.concatenate((y_train,y_part))
            else:
                y_train = y_part
    ugh=train_and_evaluate_sgd(x_train, y_train, x_test, y_test)
    model1_train[holdout]=ugh[0]
    model1_heldout[holdout]=ugh[1]
    yuck=train_and_evaluate_decision_tree(x_train, y_train, x_test, y_test)
    model2_train[holdout]=yuck[0]
    model2_heldout[holdout]=yuck[1]
    gross=train_and_evaluate_decision_stump(x_train, y_train, x_test, y_test)
    model3_train[holdout]=gross[0]
    model3_heldout[holdout]=gross[1]
    gah=train_and_evaluate_sgd_with_stumps(x_train, y_train, x_test, y_test)
    model4_train[holdout]=gah[0]
    model4_heldout[holdout]=gah[1]

    print(ugh)
    print(yuck)
    print(gross)
    print(gah)

#Test
X_train = np.load('madelon/train-X.npy')
y_train = np.load('madelon/train-y.npy')
X_test = np.load('madelon/test-X.npy')
y_test = np.load('madelon/test-y.npy')

big_ugh=train_and_evaluate_sgd(X_train, y_train, X_test, y_test)
big_yuck=train_and_evaluate_decision_tree(X_train, y_train, X_test, y_test)
big_gross=train_and_evaluate_decision_stump(X_train, y_train, X_test, y_test)
big_gah=train_and_evaluate_sgd_with_stumps(X_train, y_train, X_test, y_test)
print(big_gah)

plot_results(np.mean(model1_train),
        np.std(model1_train),
        np.mean(model1_heldout),
        np.std(model1_heldout),
        big_ugh[1],
        np.mean(model2_train),
        np.std(model2_train),
        np.mean(model2_heldout),
        np.std(model2_heldout),
        big_yuck[1],
        np.mean(model3_train),
        np.std(model3_train),
        np.mean(model3_heldout),
        np.std(model3_heldout),
        big_gross[1],
        np.mean(model4_train),
        np.std(model4_train),
        np.mean(model4_heldout),
        np.std(model4_heldout),
        big_gah[1])



