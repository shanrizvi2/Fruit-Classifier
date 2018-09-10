#import scikit modules
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.datasets import load_files

#import numpy
import numpy as np

#load image folder
fruit = load_files(r"\Users\syedi\Desktop\Proj")

#print tensor shape
images = np.array(fruit['data'])
images.shape

#put data and targets into variables
X = fruit['data']
y = fruit['target']

#split data into test and training set
X_train, X_test, y_train, y_test = train_test_split(X, y)

#standardize data
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#train model
mlp = MLPClassifier(hidden_layer_sizes=(HIDDEN LAYERS HERE))
mlp.fit(X_train, y_train)

#print report on how well the NN performed
predictions = mlp.predict(X_test)
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
