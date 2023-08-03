import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


from statistics import mode

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
greek_df = pd.read_csv("greeks.csv")

train_df.drop(columns=['Id'], inplace=True)
train_df.dropna(inplace=True)
dropped_cat = train_df.drop(columns=['EJ'])

Y = np.hsplit(dropped_cat, [-1])[1].to_numpy()
X = np.hsplit(dropped_cat, [-1])[0].to_numpy()

best_regularizations = []
regularizations = [0.0001, 0.001, 0.005, 0.01, 0.05,
                   0.1, 0.2, 0.5, 0.75, 1]

for i in range(200):
    print(i)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.15)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # PCA dimensionality reduction
    pca = PCA(n_components=20)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    # Data collection
    train_data = []
    test_data = []

    best_reg = 0
    best_score = 0

    for regularization in regularizations:
        svm = LinearSVC(C=regularization, dual=False).fit(
            X_train, Y_train.ravel())

        train_data.append(svm.score(X_train, Y_train.ravel()))

        test_score = svm.score(X_test, Y_test.ravel())
        test_data.append(test_score)
        if test_score > best_score:
            best_reg = regularization
            best_score = test_score
            print(best_score)

    best_regularizations.append(best_reg)

    # width = 0.2
    # x = np.arange(8)

    # plt.bar(x-0.2, train_data, width, color='cornflowerblue')
    # plt.bar(x, test_data, width, color='orange')
    # plt.xticks(x, regularizations)
    # plt.xlabel("L2 Regularization parameter")
    # plt.ylabel("Scores")
    # plt.legend(["Train", "Test"])
    # plt.show()


counts = [best_regularizations.count(reg) for reg in regularizations]
plt.bar([str(reg) for reg in regularizations], counts, width=0.2)
plt.xlabel("Regularization parameter")
plt.ylabel("No. of times it was the best test performer")
plt.title("Best regularization parameter for SVM")
plt.savefig('results/SVMRegularization.png')
plt.show()
# for reg in regularizations:
#     print(f"{reg} is the best regularization parameter {best_regularizations.count(reg)} times")

# print(f"The best regularization parameter is: {mode(best_regularizations)}")
