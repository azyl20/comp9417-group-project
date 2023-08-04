from sklearn import preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
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

# Finding the optimal batch size to use for Sequential model
# Was also used previously to optimise the number of nodes in hidden layer (95)
# Can also be modified to find the best optimiser during compile (adam)

i = 5
best_nodes = 0
best_score = 0
nodes = []
scores = []
while i < 50:
  nn = Sequential()
  nn.add(Dense(95, input_shape=(55,), activation='relu'))
  nn.add(Dense(55, activation='relu'))
  nn.add(Dense(1, activation='sigmoid'))
  nn.compile(optimizer='adam', metrics=['accuracy'], loss='binary_crossentropy')
  nn.fit(dropped_X_train, dropped_Y_train, epochs=200, batch_size=i)
  loss, accuracy = nn.evaluate(dropped_X_train, dropped_Y_train)
  score_test = nn.evaluate(dropped_X_test, dropped_Y_test.ravel())
  print('Number of Nodes:', i)
  print('Test loss:', score_test[0])
  print('Test accuracy:', score_test[1])
  if score_test[1] > best_score:
    best_score = score_test[1]
    best_nodes = i
  nodes.append(i)
  scores.append(score_test[1])
  i += 5

plt.plot(nodes, scores)
plt.title('Finding the optimal batch size')
plt.xlabel('batch size')
plt.ylabel('score')
plt.show()
print('best batch size:', best_nodes)
print('Highest accuracy:', best_score)
