from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import random
import itertools
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

### Pre-process data
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


# Label Encoding
train_df_bin = train_df.copy()
train_df_bin['EJ'] = train_df_bin['EJ'].apply(lambda x: 1 if x == 'B' else 0)
train_df_bin.dropna(inplace=True)

# split into X and Y
bin_Y = np.hsplit(train_df_bin, [-1])[1].to_numpy()
bin_X = np.hsplit(train_df_bin, [-1])[0].to_numpy()

# training-test split done BEFORE standardizing.
# Set random state for consistency of results
dropped_X_train, dropped_X_test, dropped_Y_train, dropped_Y_test = train_test_split(
    dropped_X, dropped_Y, test_size=0.15, random_state=123)

bin_X_train, bin_X_test, bin_Y_train, bin_Y_test = train_test_split(
    bin_X, bin_Y, test_size=0.15, random_state=123)

# using the same standardization as training dataset on test dataset to avoid data leakage
# this is done by fitting the scaler to the train df and then using it to transform the test without fitting
# no standardization required on the target variable as it is a binary classification problem
# (already 0 or 1)
scaler = StandardScaler()
dropped_X_train = scaler.fit_transform(dropped_X_train)
dropped_X_test = scaler.transform(dropped_X_test)

# Data collection
model_names = [
               "SNN - SGD",
                "SNN - ADAM",
               ]
dropped_model_train_data = []
dropped_model_test_data = []
encoded_model_train_data = []
encoded_model_test_data = []


### Sequential Model ###
for solver in ['sgd', 'adam']:
  nn = Sequential()
  nn.add(Dense(95, input_shape=(55,), activation='relu'))
  nn.add(Dense(55, activation='relu'))
  nn.add(Dense(1, activation='sigmoid'))

  nn.compile(optimizer=solver, metrics=['accuracy'], loss='binary_crossentropy')
  #nn.fit(dropped_X_train, dropped_Y_train, epochs=200, batch_size=10)

  #loss, accuracy = nn.evaluate(dropped_X_train, dropped_Y_train)
  # print('Accuracy: %.2f' % (accuracy*100))

  # # print(f"{pred}")

  nn.fit(dropped_X_train, dropped_Y_train.ravel(), epochs=200, batch_size=10)

  #     # show confusion matrix
  # train_confusion = confusion_matrix(
  #     dropped_Y_train.ravel(), (nn.predict(dropped_X_train) > 0.5).astype(int))
  # test_confusion = confusion_matrix(
  #     dropped_Y_test.ravel(), (nn.predict(dropped_X_test) > 0.5).astype(int))
  # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
  # ConfusionMatrixDisplay(confusion_matrix=train_confusion).plot(ax=axes[0])
  # ConfusionMatrixDisplay(confusion_matrix=test_confusion).plot(ax=axes[1])
  # plt.show()

  score_train = nn.evaluate(dropped_X_train, dropped_Y_train.ravel())
  score_test = nn.evaluate(dropped_X_test, dropped_Y_test.ravel())


  dropped_model_train_data.append(score_train[1])
  dropped_model_test_data.append(score_test[1])

for solver in ['sgd', 'adam']:
  nn = Sequential()
  nn.add(Dense(95, input_shape=(56,), activation='relu'))
  nn.add(Dense(56, activation='relu'))
  nn.add(Dense(1, activation='sigmoid'))

  nn.compile(optimizer=solver, metrics=['accuracy'], loss='binary_crossentropy')
  # nn.fit(bin_X_train, bin_Y_train, epochs=200, batch_size=10)

  # loss, accuracy = nn.evaluate(bin_X_train, bin_Y_train)
  # print('Accuracy: %.2f' % (accuracy*100))

  # # print(f"{pred}")

  nn.fit(bin_X_train, bin_Y_train.ravel(), epochs=200, batch_size=10)

  #     # show confusion matrix
  # train_confusion = confusion_matrix(
  #     bin_Y_train.ravel(), (nn.predict(bin_X_train) > 0.5).astype(int))
  # test_confusion = confusion_matrix(
  #     bin_Y_test.ravel(), (nn.predict(bin_X_test) > 0.5).astype(int))
  # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
  # ConfusionMatrixDisplay(confusion_matrix=train_confusion).plot(ax=axes[0])
  # ConfusionMatrixDisplay(confusion_matrix=test_confusion).plot(ax=axes[1])
  # plt.show()

  score_train = nn.evaluate(bin_X_train, bin_Y_train.ravel())
  score_test = nn.evaluate(bin_X_test, bin_Y_test.ravel())


  encoded_model_train_data.append(score_train[1])
  encoded_model_test_data.append(score_test[1])


x = np.arange(2)
width = 0.2

plt.bar(x-0.2, dropped_model_train_data, width, color='lightsteelblue')
plt.bar(x, encoded_model_train_data, width, color='bisque')
plt.bar(x+0.2, dropped_model_test_data, width, color='cornflowerblue')
plt.bar(x+0.4, encoded_model_test_data, width, color='orange')

plt.xticks(x, model_names)
plt.ylabel("Scores")
plt.legend(["Dropping (train)", "Encoding (train)",
           "Dropping (test)", "Encoding (test)"])
plt.show()
