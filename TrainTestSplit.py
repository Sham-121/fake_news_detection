#importing necessary libraries for train-test split
import pandas as pd
from sklearn.model_selection import train_test_split

#loading data
df = pd.read_csv('labeled_dataset.csv')

# get the locations
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

#splitting data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

#turning into csv


X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)

# y_train and y_test are Series, convert to DataFrame before saving
y_train.to_frame(name='label').to_csv('y_train.csv', index=False)
y_test.to_frame(name='label').to_csv('y_test.csv', index=False)

# X_train = pd.read_csv('X_train.csv')
# X_test = pd.read_csv('X_test.csv')
# y_train = pd.read_csv('y_train.csv')['label']
# y_test = pd.read_csv('y_test.csv')['label']
