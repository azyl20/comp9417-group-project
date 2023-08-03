import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


from statistics import mode

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
greek_df = pd.read_csv("greeks.csv")

train_df.drop(columns=['Id'], inplace=True)
train_df.dropna(inplace=True)
dropped_cat = train_df.drop(columns=['EJ'])

Y = np.hsplit(dropped_cat, [-1])[1].to_numpy()
X = np.hsplit(dropped_cat, [-1])[0].to_numpy()

best_ks = []
ks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

for i in range(500):
    print(i)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.15)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # Data collection
    train_data = []
    test_data = []

    best_k = 0
    best_score = 0

    for k in ks:
        knn = KNeighborsClassifier(n_neighbors=k).fit(
            X_train, Y_train.ravel())

        train_data.append(knn.score(X_train, Y_train.ravel()))

        test_score = knn.score(X_test, Y_test.ravel())
        test_data.append(test_score)
        if test_score > best_score:
            best_k = k
            best_score = test_score
            print(best_score)

    best_ks.append(best_k)

    # width = 0.2
    # x = np.arange(8)

    # plt.bar(x-0.2, train_data, width, color='cornflowerblue')
    # plt.bar(x, test_data, width, color='orange')
    # plt.xticks(x, regularizations)
    # plt.xlabel("L2 Regularization parameter")
    # plt.ylabel("Scores")
    # plt.legend(["Train", "Test"])
    # plt.show()


counts = [best_ks.count(k) for k in ks]
plt.bar([str(k) for k in ks], counts, width=0.2)
plt.xlabel("No. of neighbours")
plt.ylabel("No. of times it was the best test performer")
plt.title("Best k for KNN")
plt.show()
# for reg in regularizations:
#     print(f"{reg} is the best regularization parameter {best_regularizations.count(reg)} times")

# print(f"The best regularization parameter is: {mode(best_regularizations)}")
