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
print('best batch size:', best_nodes)
print('Highest accuracy:', best_score)
