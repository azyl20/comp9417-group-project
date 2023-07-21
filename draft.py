import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import GradientBoostingClassifier

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
greek_df = pd.read_csv("greeks.csv")

no_condition = train_df[train_df['Class'] == 0]
has_condition = train_df[train_df['Class'] == 1]

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

# converting categorical feature EJ into binary feature
train_df_bin = train_df.copy()
train_df_bin['EJ'] = train_df_bin['EJ'].apply(lambda x: 1 if x == 'B' else 0)
train_df_bin.dropna(inplace=True)

# split into X and Y
bin_Y = np.hsplit(train_df_bin, [-1])[1].to_numpy()
bin_X = np.hsplit(train_df_bin, [-1])[0].to_numpy()

# standardising and scaling Xs
scaler = StandardScaler()
dropped_X = scaler.fit_transform(dropped_X)
bin_X = scaler.fit_transform(bin_X)

############## LOGISTIC REGRESSION ##############

clf_dropped = LogisticRegression(
    random_state=0).fit(dropped_X, dropped_Y.ravel())

dropped_Y_pred = clf_dropped.predict(dropped_X)
confusion_dropped = confusion_matrix(dropped_Y, dropped_Y_pred)
disp_dropped = ConfusionMatrixDisplay(confusion_matrix=confusion_dropped)

print("score when dropping categorical variables:")
print(clf_dropped.score(dropped_X, dropped_Y.ravel()))
disp_dropped.plot()
plt.show()

#-----------------------#
clf_bin = LogisticRegression(random_state=0).fit(bin_X, bin_Y.ravel())

bin_Y_pred = clf_bin.predict(bin_X)
confusion_bin = confusion_matrix(bin_Y, bin_Y_pred)
disp_bin = ConfusionMatrixDisplay(confusion_matrix=confusion_bin)

print("score when encoding categorical variables:")
print(clf_bin.score(bin_X, bin_Y.ravel()))
disp_bin.plot()
plt.show()

############## GRADIENT BOOSTING CLASSIFIER ##############
gradboost_dropped = GradientBoostingClassifier(
    n_estimators=100, learning_rate=0.1, max_depth=2, random_state=0).fit(dropped_X, dropped_Y.ravel())

dropped_Y_pred = gradboost_dropped.predict(dropped_X)
confusion_dropped = confusion_matrix(dropped_Y, dropped_Y_pred)
disp_dropped = ConfusionMatrixDisplay(confusion_matrix=confusion_dropped)

print("score when dropping categorical variables using GradientBoostingClassifier:")
print(gradboost_dropped.score(dropped_X, dropped_Y.ravel()))
disp_dropped.plot()
plt.show()

#-----------------------#
gradboost_bin = GradientBoostingClassifier(
    n_estimators=100, learning_rate=0.1, max_depth=2, random_state=0).fit(bin_X, bin_Y.ravel())

bin_Y_pred = gradboost_bin.predict(bin_X)
confusion_bin = confusion_matrix(bin_Y, bin_Y_pred)
disp_bin = ConfusionMatrixDisplay(confusion_matrix=confusion_bin)

print("score when encoding categorical variables using GradientBoostingClassifier:")
print(gradboost_bin.score(bin_X, bin_Y.ravel()))
disp_bin.plot()
plt.show()
