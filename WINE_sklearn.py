import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category="UndefinedMetricWarning")
warnings.simplefilter(action='ignore', category="DataConversionWarning")

from Data.data import get_data, one_dim_array
from UTILS.utils import quality
from sklearn.neural_network import MLPClassifier as mlp
from sklearn.preprocessing import LabelEncoder as LE
from sklearn.metrics import classification_report

X,y = get_data(sample_replicats=100000)
#y = one_dim_array(y)

model = mlp(learning_rate='adaptive',hidden_layer_sizes=(20,10),
           max_iter=1000000,verbose=1)

le = LE()
le.fit(y)
print('classes: {i}'.format(i=le.classes_))



model.fit(X,y)
y_pred = model.predict(X)
y_pred = le.transform(y_pred)


print(classification_report(y,y_pred))
