import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
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

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15)

# standardising and scaling Xs
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

############## RANDOM FOREST ###############

forest = RandomForestClassifier(
    n_estimators=400, min_samples_split=2, min_samples_leaf=1, max_depth=None, random_state=123)

############## LOGISTIC REGRESSION ###############
lgclassifier = LogisticRegression(random_state=123)

############## GRADIENT BOOSTING CLASSIFIER ##############
gradboost = HistGradientBoostingClassifier(max_depth=2, random_state=123)

############## K NEIGHBOURS ##############
knn = KNeighborsClassifier(n_neighbors=3)

############## STACKING ##############
estimators = [
    ('rf', forest),
    ('boost', gradboost),
    ('knn', knn),
]

stack_classifier = StackingClassifier(
    estimators=estimators, final_estimator=lgclassifier)

stack_classifier.fit(X_train, Y_train.ravel())
print(
    f"\nStacking classifier training Accuracy: {stack_classifier.score(X_train, Y_train)}")
print(
    f"\nStacking classifier testing Accuracy: {stack_classifier.score(X_test, Y_test)}")

# print(stack_classifier.predict_proba(X_test))

forest.fit(X_train, Y_train.ravel())
print(
    f"\nForest classifier training Accuracy: {forest.score(X_train, Y_train)}")
print(
    f"\nForest classifier testing Accuracy: {forest.score(X_test, Y_test)}")

# print(forest.predict_proba(X_test))

gradboost.fit(X_train, Y_train.ravel())
print(
    f"\nHistGradientBoosting classifier training Accuracy: {gradboost.score(X_train, Y_train)}")

print(
    f"\nHistGradientBoosting classifier testing Accuracy: {gradboost.score(X_test, Y_test)}")

knn.fit(X_train, Y_train.ravel())
print(
    f"\nK Nearest Neighbours classifier training Accuracy: {knn.score(X_train, Y_train)}")

print(
    f"\nK Nearest Neighbours classifier testing Accuracy: {knn.score(X_test, Y_test)}")

# print(gradboost.predict_proba(X_test))

# submission = {}
# submission["class_0"] = Y_pred[:, 0]
# submission["class_1"] = Y_pred[:, 1]

# submission_df = pd.DataFrame(submission, index=test_df.index)

# print(submission_df)

# submission_df.to_csv('submission.csv')
