import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from lib.binary_grey_wolf import BinaryGreyWolfOptimizer
from lib.estimator import LogError


def german(mode='bgwo'):
    path = r'raw_data/test_data_01/german_credit.csv'
    data = pd.read_csv(path)
    y_train = data['Creditability']
    X_train = data.drop('Creditability', axis=1)
    del data
    logerror = LogError(5, 5)
    bwo = BinaryGreyWolfOptimizer(X_train, y_train, N=10, T=30, estimator=logerror)
    if mode == 'bwo':
        X_feature_index = bwo.bgwo()
    else:
        X_feature_index = bwo.ibgwo()
    print(X_feature_index)


def aus(mode='bgwo'):
    path = r'raw_data/test_data_01/aus.csv'
    data = pd.read_csv(path)
    y_train = data['y']
    X_train = data.drop('y', axis=1)
    del data

    logerror = LogError(5, 5)

    bwo = BinaryGreyWolfOptimizer(X_train, y_train, N=10, T=30, estimator=logerror)
    if mode == 'bwo':
        X_feature_index = bwo.bgwo()
    else:
        X_feature_index = bwo.ibgwo()
    print(X_feature_index)


if __name__ == '__main__':
    german('bwo')
    # german('ibgwo')
    # aus('bgwo')
    # aus('ibgwo')
    pass
