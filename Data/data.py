import numpy as np
import pandas as pd
from sklearn.datasets import load_wine


def get_data(sample_replicats=1000,as_multi_class=False):
    #print(sample_replicats.__class__)
    assert sample_replicats.__class__ != 'int',"Keinen Integer eingegeben"
    data = load_wine()

    X = []
    y = []
    for i in range(sample_replicats):
        X.append(data.data)
        y.append(data.target)
    X = tuple(X)
    y_temp = tuple(y)




    X = np.vstack(X)
    y = np.hstack(y_temp)
    if(as_multi_class==False):
        X = pd.DataFrame(data=X, columns=data.feature_names)
        return (X,y)
    else:
        y = np.zeros((len(y), 3))
        for k, i in enumerate(y_temp[0]):
            if i == 0:
                y[k, 0] = 1
            elif i == 1:
                y[k, 1] = 1
            elif i == 2:
                y[k, 2] = 1
            else:
                print('class not avalable')


        X = pd.DataFrame(data=X, columns=data.feature_names)
        return(X,y)



def one_dim_array(y=None):
    row,col = y.shape
    y_new = np.zeros((row))
    for r in range(row):
        for c in range(col):
            if y[r,c] == 1:
                y_new[r] = int(c)
    return y_new