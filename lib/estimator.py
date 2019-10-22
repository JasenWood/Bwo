"""
https://blog.csdn.net/qq_32590631/article/details/82831613
https://www.baidu.com/link?url=oekx7bhx4hdDxoXwoCizLqJp6p2i-vu2QgUYdHbLcDEgQTu8QdzoXpj8VKhNPw5a8VOid1vQmupbB_73QDtv5_&wd=&eqid=ae26d5a5000897cd000000065daeecf5
https://blog.csdn.net/tanzuozhev/article/details/79109311
"""
"""暂时支持accuracy，f1，roc_auc均为正指标"""
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier


class KnnError:
    def __init__(self, k, k_fold):
        self.k = k
        self.cv = StratifiedKFold(k_fold, shuffle=True)

    def get_error(self, X_train, y_train, X, error_name):
        X_train = X_train.iloc[:, X == 1]
        clf = KNeighborsClassifier(n_neighbors=self.k, weights='distance')
        score = cross_val_score(clf, X_train, y_train, cv=self.cv, scoring=error_name).mean()
        return -score


class LogError:
    def __init__(self, k, k_fold):
        self.k = k
        self.cv = StratifiedKFold(k_fold, shuffle=True)

    def get_error(self, X_train, y_train, X, error_name):
        X_train = X_train.iloc[:, X == 1]
        clf = LogisticRegression(solver='liblinear')
        score = cross_val_score(clf, X_train, y_train, cv=self.cv, scoring=error_name).mean()
        return -score


class RfError:
    def __init__(self, k, k_fold):
        self.k = k
        self.cv = StratifiedKFold(k_fold, shuffle=True)

    def get_error(self, X_train, y_train, X, error_name):
        """http://www.sohu.com/a/280496452_163476"""
        X_train = X_train.iloc[:, X == 1]
        clf = RandomForestClassifier(oob_score=True, random_state=10)
        score = cross_val_score(clf, X_train, y_train, cv=self.cv, scoring=error_name).mean()
        return -score


if __name__ == '__main__':
    X_train = pd.read_csv('raw_data/test_data/X_train.csv', header=None)
    y_train = pd.read_csv('raw_data/test_data/y_train.csv', header=None).iloc[:, 0]
    knn = KnnError(5, 2)
    knn.get_error(X_train, y_train, X=np.ones(shape=34), error_name='roc_auc')
