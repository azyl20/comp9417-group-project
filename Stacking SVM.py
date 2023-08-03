import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier


from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv", index_col=0)
greek_df = pd.read_csv("greeks.csv")

# removing Id column and Nan
train_df.drop(columns=['Id'], inplace=True)
train_df.dropna(inplace=True)
test_df.dropna(inplace=True)

# dropping 'EJ' which is a categorical feature
train_df.drop(columns=['EJ'], inplace=True)
test_df.drop(columns=['EJ'], inplace=True)


# split into X and Y
Y = np.hsplit(train_df, [-1])[1].to_numpy()
X = np.hsplit(train_df, [-1])[0].to_numpy()

stack_train = []
stack_test = []
forest_train = []
forest_test = []
boost_train = []
boost_test = []
svm_train = []
svm_test = []


for i in range(200):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15)
    # standardising and scaling training Xs
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    ############## RANDOM FOREST ###############

    forest = RandomForestClassifier(n_estimators=100)

    ############## LOGISTIC REGRESSION ###############
    lgclassifier = LogisticRegression()

    ############## GRADIENT BOOSTING CLASSIFIER ##############
    gradboost = HistGradientBoostingClassifier(max_depth=2)

    ############## SUPPORT VECTOR MACHINE ##############
    svm = LinearSVC(C=0.001, dual=False)

    ############## STACKING ##############
    estimators = [
        ('rf', forest),
        ('boost', gradboost),
        ('svm', svm),
    ]
    stack_classifier = StackingClassifier(
        estimators=estimators, final_estimator=lgclassifier)
    stack_classifier.fit(X_train, Y_train.ravel())
    train_score = stack_classifier.score(X_train, Y_train)
    stack_train.append(train_score)
    test_score = stack_classifier.score(X_test, Y_test)
    stack_test.append(test_score)
    # print(
    #     f"\nStacking classifier training Accuracy: {train_score}")
    # print(
    #     f"\nStacking classifier testing Accuracy: {test_score}")

    # print(stack_classifier.predict_proba(X_test))

    forest.fit(X_train, Y_train.ravel())
    train_score = forest.score(X_train, Y_train)
    forest_train.append(train_score)
    test_score = forest.score(X_test, Y_test)
    forest_test.append(test_score)
    # print(
    #     f"\nForest classifier training Accuracy: {train_score}")
    # print(
    #     f"\nForest classifier testing Accuracy: {test_score}")

    # print(forest.predict_proba(X_test))

    gradboost.fit(X_train, Y_train.ravel())
    train_score = gradboost.score(X_train, Y_train)
    boost_train.append(train_score)
    test_score = gradboost.score(X_test, Y_test)
    boost_test.append(test_score)
    # print(
    #     f"\nHistGradientBoosting classifier training Accuracy: {train_score}")

    # print(
    #     f"\nHistGradientBoosting classifier testing Accuracy: {test_score}")

    svm.fit(X_train, Y_train.ravel())
    train_score = svm.score(X_train, Y_train)
    svm_train.append(train_score)
    test_score = svm.score(X_test, Y_test)
    svm_test.append(test_score)
    # print(
    #     f"\nSVM training Accuracy: {train_score}")

    # print(
    #     f"\nSVM testing Accuracy: {test_score}")

labels = ['stack_train', 'forest_train',
          'boost_train', 'svm_train']

i = 0
for data in [stack_train, forest_train, boost_train, svm_train]:
    plt.scatter(list(range(200)), data, label=labels[i], marker='.')
    i += 1

plt.xlabel('iterations')
plt.ylabel('score')
plt.title('training scores')
plt.legend()
plt.show()

labels = ['stack_test', 'forest_test',
          'boost_test', 'svm_test']
i = 0
for data in [stack_test, forest_test, boost_test, svm_test]:
    plt.scatter(list(range(200)), data, label=labels[i], marker='.')
    i += 1

plt.xlabel('iterations')
plt.ylabel('score')
plt.title('test scores')
plt.legend()
plt.show()

print(f"Stacking test average: {sum(stack_test)/len(stack_test)}")
print(f"Random Forest test average: {sum(forest_test)/len(forest_test)}")
print(f"gradient boosting test average: {sum(boost_test)/len(boost_test)}")
print(f"SVM test average: {sum(svm_test)/len(svm_test)}")

fig, axs = plt.subplots(2, 2)
axs[0, 0].boxplot(stack_test)
axs[0, 0].set_title('Stacking Scores')
axs[0, 1].boxplot(forest_test)
axs[0, 1].set_title('Random Forest Scores')
axs[1, 0].boxplot(boost_test)
axs[1, 0].set_title('HistGBC scores')
axs[1, 1].boxplot(svm_test)
axs[1, 1].set_title('SVM scores')

for i in range(2):
    for j in range(2):
        axs[i, j].set_ylim(0.8, 1)
        axs[i, j].xaxis.set_visible(False)

plt.show()


# print(gradboost.predict_proba(X_test))

# submission = {}
# submission["class_0"] = Y_pred[:, 0]
# submission["class_1"] = Y_pred[:, 1]

# submission_df = pd.DataFrame(submission, index=test_df.index)

# print(submission_df)

# submission_df.to_csv('submission.csv')
