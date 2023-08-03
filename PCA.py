import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

import random

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
greek_df = pd.read_csv("greeks.csv")

# removing Id column
train_df.drop(columns=['Id'], inplace=True)

# dropping 'EJ' which is a categorical feature
train_df_dropped_cat = train_df.drop(columns=['EJ'])
train_df_dropped_cat.dropna(inplace=True)
# print(train_df_dropped_cat.isnull().any())
print(np.shape(train_df_dropped_cat))

# split into X and Y
dropped_Y = np.hsplit(train_df_dropped_cat, [-1])[1].to_numpy()
dropped_X = np.hsplit(train_df_dropped_cat, [-1])[0].to_numpy()
# print(dropped_X)
# print(dropped_Y)

model_names = ["Log. Regression", "KNN", "Random Forest", "SVM",
               "HGBC depth 2"]
num_times_better = [0, 0, 0, 0, 0]


def displayScreePlot():
    dropped_X_train, dropped_X_test, dropped_Y_train, dropped_Y_test = train_test_split(
        dropped_X, dropped_Y, test_size=0.15)

    # using the same standardization as training dataset on test dataset to avoid data leakage
    # this is done by fitting the scaler to the train df and then using it to transform the test without fitting
    # no standardization required on the target variable as it is a binary classification problem
    # (already 0 or 1)
    scaler = StandardScaler()
    dropped_X_train = scaler.fit_transform(dropped_X_train)
    dropped_X_test = scaler.transform(dropped_X_test)

    # PCA dimensionality reduction
    pca = PCA()
    reduced_X_train = pca.fit_transform(dropped_X_train)
    reduced_X_test = pca.transform(dropped_X_test)

    # Show Scree Plot and variance ratio matrix
    print(pca.explained_variance_ratio_)
    eigenvalues = pca.explained_variance_
    prop_var = eigenvalues / np.sum(eigenvalues)

    plt.plot(np.arange(1, len(eigenvalues)+1),
             eigenvalues, marker='o')
    plt.xlabel('Principal Component')
    plt.ylabel('Eigenvalue')
    plt.title('Scree Plot for Eigenvalues')
    plt.axhline(y=1, color='r',
                linestyle='--')
    plt.grid(True)
    plt.savefig('results/ScreePlotPCA.png')
    plt.show()


# # Uncomment to see Scree Plot
# displayScreePlot()


# training-test split done BEFORE standardizing.
# Set random state for consistency of results
for i in range(500):
    dropped_X_train, dropped_X_test, dropped_Y_train, dropped_Y_test = train_test_split(
        dropped_X, dropped_Y, test_size=0.15)

    # using the same standardization as training dataset on test dataset to avoid data leakage
    # this is done by fitting the scaler to the train df and then using it to transform the test without fitting
    # no standardization required on the target variable as it is a binary classification problem
    # (already 0 or 1)
    scaler = StandardScaler()
    dropped_X_train = scaler.fit_transform(dropped_X_train)
    dropped_X_test = scaler.transform(dropped_X_test)

    # PCA dimensionality reduction
    pca = PCA(n_components=17)
    reduced_X_train = pca.fit_transform(dropped_X_train)
    reduced_X_test = pca.transform(dropped_X_test)
    reduced_Y_train = dropped_Y_train
    reduced_Y_test = dropped_Y_test

    # Data collection

    dropped_model_train_data = []
    dropped_model_test_data = []
    pca_model_train_data = []
    pca_model_test_data = []

    j = 0
    for model in [LogisticRegression(random_state=0),
                  KNeighborsClassifier(n_neighbors=3),
                  RandomForestClassifier(),
                  LinearSVC(C=0.001, dual=False),
                  HistGradientBoostingClassifier(max_depth=2, random_state=0)]:
        # Fit model to training data
        model.fit(dropped_X_train, dropped_Y_train.ravel())

        # show confusion matrix
        train_confusion = confusion_matrix(
            dropped_Y_train.ravel(), model.predict(dropped_X_train))
        test_confusion = confusion_matrix(
            dropped_Y_test.ravel(), model.predict(dropped_X_test))

        # Collect data for plotting
        dropped_model_train_data.append(model.score(
            dropped_X_train, dropped_Y_train.ravel()))
        dropped_model_test_data.append(model.score(
            dropped_X_test, dropped_Y_test.ravel()))

        j += 1

    j = 0
    for model in [LogisticRegression(random_state=0),
                  KNeighborsClassifier(n_neighbors=3),
                  RandomForestClassifier(),
                  LinearSVC(C=0.001, dual=False),
                  HistGradientBoostingClassifier(max_depth=2, random_state=0)]:
        # Fit model to training data
        model.fit(reduced_X_train, reduced_Y_train.ravel())

        # show confusion matrix
        train_confusion = confusion_matrix(
            reduced_Y_train.ravel(), model.predict(reduced_X_train))
        test_confusion = confusion_matrix(
            reduced_Y_test.ravel(), model.predict(reduced_X_test))

        # Collect data for plotting
        pca_model_train_data.append(model.score(
            reduced_X_train, reduced_Y_train.ravel()))
        pca_model_test_data.append(model.score(
            reduced_X_test, reduced_Y_test.ravel()))

        j += 1

    for k, value in enumerate(dropped_model_test_data):
        if pca_model_test_data[k] > value:
            num_times_better[k] += 1

    print(i)

plt.bar(model_names, num_times_better)
plt.title("No. times PCA improved test performance")
plt.savefig("results/PCAperf.png")
plt.show()
