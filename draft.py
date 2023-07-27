import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split


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

dropped_X_train, dropped_X_test, dropped_Y_train, dropped_Y_test = train_test_split(
    dropped_X, dropped_Y, test_size=0.15, random_state=123)

bin_X_train, bin_X_test, bin_Y_train, bin_Y_test = train_test_split(
    bin_X, bin_Y, test_size=0.15, random_state=123)

dropped_cat = train_df.drop(columns=['EJ'])
Y = np.hsplit(dropped_cat, [-1])[1].to_numpy()
X = np.hsplit(dropped_cat, [-1])[0].to_numpy()
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.15, random_state=123)


# Data collection
model_names = ["Log. Regression",
               "GBC depth 1",
               "GBC depth 2",
               "HGBC depth 1",
               "HGBC depth 2"]
dropped_model_train_data = []
dropped_model_test_data = []
encoded_model_train_data = []
encoded_model_test_data = []


i = 0
for model in [LogisticRegression(random_state=0),
              GradientBoostingClassifier(
                  n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0),
              GradientBoostingClassifier(
                  n_estimators=100, learning_rate=0.1, max_depth=2, random_state=0),
              HistGradientBoostingClassifier(max_depth=1, random_state=0),
              HistGradientBoostingClassifier(max_depth=2, random_state=0)]:
    # Fit model to training data
    model.fit(dropped_X_train, dropped_Y_train.ravel())

    # show confusion matrix
    train_confusion = confusion_matrix(
        dropped_Y_train.ravel(), model.predict(dropped_X_train))
    test_confusion = confusion_matrix(
        dropped_Y_test.ravel(), model.predict(dropped_X_test))

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    axes[0].set_title(
        f"Training Confusion Matrix, Score: {model.score(dropped_X_train, dropped_Y_train.ravel()):.4f}")
    ConfusionMatrixDisplay(confusion_matrix=train_confusion).plot(ax=axes[0])
    axes[1].set_title(
        f"Testing Confusion Matrix, Score: {model.score(dropped_X_test, dropped_Y_test.ravel()):.4f}")
    ConfusionMatrixDisplay(confusion_matrix=test_confusion).plot(ax=axes[1])
    fig.suptitle(f"{model_names[i]} - dropping categorical vars.")
    plt.show()

    # Collect data for plotting
    dropped_model_train_data.append(model.score(
        dropped_X_train, dropped_Y_train.ravel()))
    dropped_model_test_data.append(model.score(
        dropped_X_test, dropped_Y_test.ravel()))

    i += 1

i = 0
for model in [LogisticRegression(random_state=0),
              GradientBoostingClassifier(
                  n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0),
              GradientBoostingClassifier(
                  n_estimators=100, learning_rate=0.1, max_depth=2, random_state=0),
              HistGradientBoostingClassifier(max_depth=1, random_state=0),
              HistGradientBoostingClassifier(max_depth=2, random_state=0)]:
    # Fit model to training data
    model.fit(bin_X_train, bin_Y_train.ravel())

    # show confusion matrix
    train_confusion = confusion_matrix(
        bin_Y_train.ravel(), model.predict(bin_X_train))
    test_confusion = confusion_matrix(
        bin_Y_test.ravel(), model.predict(bin_X_test))

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    axes[0].set_title(
        f"Training Confusion Matrix, Score: {model.score(bin_X_train, bin_Y_train.ravel()):.4f}")
    ConfusionMatrixDisplay(confusion_matrix=train_confusion).plot(ax=axes[0])
    axes[1].set_title(
        f"Testing Confusion Matrix, Score: {model.score(bin_X_test, bin_Y_test.ravel()):.4f}")
    ConfusionMatrixDisplay(confusion_matrix=test_confusion).plot(ax=axes[1])
    fig.suptitle(f"{model_names[i]} - encoding categorical vars.")
    plt.show()

    # Collect data for plotting
    encoded_model_train_data.append(model.score(
        bin_X_train, bin_Y_train.ravel()))
    encoded_model_test_data.append(model.score(
        bin_X_test, bin_Y_test.ravel()))

    i += 1


x = np.arange(5)
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


with_na_gbc = HistGradientBoostingClassifier(max_depth=2, random_state=0)
with_na_gbc.fit(X_train, Y_train.ravel())

plt.bar(["Train", "Test"], [with_na_gbc.score(X_train, Y_train.ravel()),
        with_na_gbc.score(X_test, Y_test.ravel())], color='aquamarine')

plt.title(
    "Training and Test Scores of\nHistGradientBoostingClassifier\non original dataset")
plt.ylabel("Scores")
plt.yticks(ticks=np.arange(0, 1.05, step=0.05))
plt.show()
