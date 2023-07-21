import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import HistGradientBoostingClassifier

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
greek_df = pd.read_csv("greeks.csv")

# removing Id column
train_df.drop(columns=['Id'], inplace=True)
test_df.drop(columns=['Id'], inplace=True)

# dropping 'EJ' which is a categorical feature
train_df.drop(columns=['EJ'], inplace=True)
test_df.drop(columns=['EJ'], inplace=True)


# split into X and Y
Y_train = np.hsplit(train_df, [-1])[1].to_numpy()
X_train = np.hsplit(train_df, [-1])[0].to_numpy()
X_test = test_df


# standardising and scaling Xs
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

############## GRADIENT BOOSTING CLASSIFIER ##############
gradboost = HistGradientBoostingClassifier().fit(X_train, Y_train.ravel())

Y_pred_train = gradboost.predict(X_train)
confusion = confusion_matrix(Y_train, Y_pred_train)
disp = ConfusionMatrixDisplay(confusion_matrix=confusion)

print("Train Score:")
print(gradboost.score(X_train, Y_train.ravel()))
disp.plot()
plt.show()

# Y_pred = gradboost.predict(X_test)
# confusion_test = confusion_matrix(Y_test, Y_pred)
# disp_test = ConfusionMatrixDisplay(confusion_matrix=confusion_test)

# print("Test Score:")
# print(gradboost.score(X_test, Y_test.ravel()))
# disp_test.plot()
# plt.show()
