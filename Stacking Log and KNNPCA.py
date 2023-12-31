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
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

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
log_train = []
log_test = []
knn_train = []
knn_test = []

for i in range(50):
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
    gradboost = HistGradientBoostingClassifier(max_depth=3)

    ############## Log. Regression ##############
    log = make_pipeline(FunctionTransformer(lambda X: PCA(
        n_components=20).fit_transform(X)), LogisticRegression(C=0.1))

    ############## K NEIGHBOURS ##############
    # creating a pipeline to pass in a PCA reduced dataset to the svm learner
    # n_components parameter optimised in PCA.py
    # source: https://stats.stackexchange.com/questions/540800/stackingclassifier-different-feature-sets

    knn = make_pipeline(FunctionTransformer(lambda X: PCA(
        n_components=20).fit_transform(X)), KNeighborsClassifier(n_neighbors=3))

    ############## STACKING ##############
    estimators = [
        ('rf', forest),
        ('boost', gradboost),
        ('log', log),
        ('knn', knn)
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

    log.fit(X_train, Y_train.ravel())
    train_score = log.score(X_train, Y_train)
    log_train.append(train_score)
    test_score = log.score(X_test, Y_test)
    log_test.append(test_score)
    # print(
    #     f"\nK Nearest Neighbours classifier training Accuracy: {train_score}")

    # print(
    #     f"\nK Nearest Neighbours classifier testing Accuracy: {test_score}")
    knn.fit(X_train, Y_train.ravel())
    train_score = knn.score(X_train, Y_train)
    knn_train.append(train_score)
    test_score = knn.score(X_test, Y_test)
    knn_test.append(test_score)

labels = ['stack_train', 'forest_train',
          'boost_train', 'log_train', 'knn_train']

i = 0
for data in [stack_train, forest_train, boost_train, log_train, knn_train]:
    plt.scatter(list(range(50)), data, label=labels[i], marker='.')
    i += 1

plt.xlabel('iterations')
plt.ylabel('score')
plt.title('training scores')
plt.legend()
plt.show()

labels = ['stack_test', 'forest_test',
          'boost_test', 'log_test', 'knn_test']
i = 0
for data in [stack_test]:
    plt.scatter(list(range(50)), data, label=labels[i], marker='.')
    i += 1

plt.xlabel('iterations')
plt.ylabel('score')
plt.title('test scores')
plt.legend()
plt.show()

print(f"Stacking test average: {sum(stack_test)/len(stack_test)}")
print(f"Random Forest test average: {sum(forest_test)/len(forest_test)}")
print(f"gradient boosting test average: {sum(boost_test)/len(boost_test)}")
print(f"Log. regression test average: {sum(log_test)/len(log_test)}")
print(f"KNN test average: {sum(knn_test)/len(knn_test)}")

plt.boxplot(stack_test)
plt.title('Stacking Scores')
plt.show()


# print(gradboost.predict_proba(X_test))

# submission = {}
# submission["class_0"] = Y_pred[:, 0]
# submission["class_1"] = Y_pred[:, 1]

# submission_df = pd.DataFrame(submission, index=test_df.index)

# print(submission_df)

# submission_df.to_csv('submission.csv')
