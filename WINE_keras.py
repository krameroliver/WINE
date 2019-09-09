#warnings.simplefilter(action='ignore', category=FutureWarning)
#warnings.simplefilter(action='ignore', category=RuntimeWarning)
#warnings.simplefilter(action='ignore', category="UndefinedMetricWarning")
#warnings.simplefilter(action='ignore', category="DataConversionWarning")
from keras import backend as K

from Data.data import get_data

#K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=2, inter_op_parallelism_threads=2)))

from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import numpy as np

X,y = get_data(sample_replicats=100,as_multi_class=True)

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
ann.add(Dense(2, activation='sigmoid', input_shape=(11,)))
# Add one hidden layer
ann.add(Dense(1, activation='sigmoid'))
# Add an output layer
ann.add(Dense(3, activation='sigmoid'))

ann.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

ann.fit(X_train, y_train, epochs=200, batch_size=1,verbose=1)



y_pred = ann.predict(X_test)
y_pred_classes = np.zeros_like(y_pred)
for r in range(y_pred_classes.shape[0]):
  i, j = np.unravel_index(y_pred.argmax(axis=0), y_pred.shape)
  y_pred_classes[i,j]=1



print(y_test)
print(y_pred_classes)

#print(confusion_matrix(y_test, y_pred))
#print(f1_score(y_test,y_pred))