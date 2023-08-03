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
from math import exp
from sklearn.metrics import accuracy_score
from numpy.random import randn
from numpy.random import rand

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


## Multi-Layer Perceptron Functions ##

# activation function
def activate(layer, weights):
	activation = weights[-1]
	for i in range(len(layer)):
		activation += weights[i] * layer[i]
	return activation

# activation function for a network
def layer_predict(row, network):
	new_inputs = row
	# Iterate through layers, then nodes
	for layer in network:
		outputs = []
		# activate each node in the layer, and run it through the transfer function
		for node in layer:
			activation = activate(new_inputs, node)
			output = 1.0 / (1.0 + exp(-activation))
			outputs.append(output)
		# In a multi-layer perceptron
		# The second layer's inputs are retrieved from the first layer's outputs
		# And so on
		new_inputs = outputs
	return new_inputs[0]

# Predict for each layer in the network
# characteristics is essentially the X dataset
def predict_dataset(characteristics, network):
	predictions = []
	for row in characteristics:
		predictions.append(layer_predict(row, network))
	return predictions

# objective function
# characteristics is essentially the X dataset
# conditions is essentially the y dataset
def objective(characteristics, conditions, network):
	pred = [round(i) for i in predict_dataset(characteristics, network)]
	return accuracy_score(conditions, pred)

# Sotchastic Hill Climbing
def hillclimbing(X, y, objective, best, n_iter, step_size):
	iterations = []
	evals = []
	best_accuracy = objective(X, y, best)
	for i in range(n_iter):
		# take a step in the search network
		new_net = []
		for layer in best:
			new_layer = []
			for node in layer:
				mod = step_size * randn(len(node))
				new_node = mod + node.copy()
				new_layer.append(new_node)
			new_net.append(new_layer)
		current = new_net
		current_accuracy = objective(X, y, current)
		if current_accuracy >= best_accuracy:
			best, best_accuracy = current, current_accuracy
			iterations.append(i)
			print('Iteration %d New Score %f' % (i, best_accuracy))
			evals.append(current_accuracy)
  # plotting the life cycle of the model

  # plt.plot(iterations, evals)
  # plt.title('optimisation of MLP weights by Iteration')
  # plt.xlabel('Iteration')
  # plt.ylabel('Accuracy on Train Dataset')
	return [best, best_accuracy]


# In increments of 50, the MLP runs i iterations
# We can then see through the graph which value of i produces the best outcome
i = 50
n_iters = []
accuracies = []
while i < 2000:
  n_iter = i
  step_size = 0.1
  num_characteristics = dropped_X.shape[1]
  # one hidden layer and an output layer
  hidden_nodes = 10
  init_hidden = [rand(num_characteristics + 1) for _ in range(0, 10)]
  init_output = [rand(hidden_nodes + 1)]
  network = [init_hidden, init_output]
  # commence hill climbing
  network, accuracy = hillclimbing(dropped_X_train, dropped_Y_train, objective, network, n_iter, step_size)
  print('Training Data Highest Accuracy: %f' % (accuracy))
  pred = [round(pred) for pred in predict_dataset(dropped_X_test, network)]
  accuracy = accuracy_score(dropped_Y_test, pred)
  print('Test Accuracy: %.5f' % (accuracy * 100))

  n_iters.append(i)
  accuracies.append(accuracy)

  i += 50
plt.plot(n_iters, accuracies)
plt.title("Finding the best number of iterations for a high score")
plt.xlabel('Number of iterations')
plt.ylabel('Score')
