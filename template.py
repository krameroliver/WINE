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
from sklearn.model_selection import train_test_split



X,y = get_data(sample_replicats=1000,as_multi_class=False)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=42)


