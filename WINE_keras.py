import warnings
#warnings.simplefilter(action='ignore', category=FutureWarning)
#warnings.simplefilter(action='ignore', category=RuntimeWarning)
#warnings.simplefilter(action='ignore', category="UndefinedMetricWarning")
#warnings.simplefilter(action='ignore', category="DataConversionWarning")
from Data.data import get_data

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score
from sklearn.preprocessing import LabelEncoder as LE

import numpy as np

from keras.wrappers.scikit_learn import KerasClassifier

X,y = get_data(sample_replicats=10,as_multi_class=True)

#le = LE()
#le.fit(y)
#print("classes: {i}".format(i=le.classes_))
X=X.ix[:,0:11]
print(X.shape)
print(y.shape)
r,c = X.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=42)

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

ann = Sequential()
ann.add(Dense(64, activation='relu', input_shape=(11,)))
# Add one hidden layer
ann.add(Dense(80, activation='relu'))
# Add an output layer
ann.add(Dense(3, activation='sigmoid'))

ann.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

ann.fit(X_train, y_train, epochs=20, batch_size=1,verbose=0)



y_pred = ann.predict(X_test)
y_pred_classes = np.zeros_like(y_pred)
for r in range(y_pred_classes.shape[0]):
  i, j = np.unravel_index(y_pred.argmax(axis=0), y_pred.shape)
  y_pred_classes[i,j]=1



#print(y_test)
#print(y_pred_classes)

print(confusion_matrix(y_test, y_pred))
print(f1_score(y_test,y_pred))