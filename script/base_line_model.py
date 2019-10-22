# -*- coding: utf-8 -*-

import pandas as pd
import pandas_profiling
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV


def logistic_score_report(estimator, X_train, y_train, cv, name='estimator'):
    auc = cross_val_score(estimator, X_train, y_train, scoring='roc_auc', cv=cv).mean()
    accuracy = cross_val_score(estimator, X_train, y_train, scoring='accuracy', cv=cv).mean()
    recall = cross_val_score(estimator, X_train, y_train, scoring='recall', cv=cv).mean()
    print("{}: auc:{:f}, recall:{:f}, accuracy:{:f}".format(name, auc, recall, accuracy))


def german():
    path = r'raw_data/test_data_01/german_credit.csv'
    data = pd.read_csv(path)
    # profile = data.profile_report(title='German Data')
    # profile.to_file(output_file="Training_Data.html")
    # del profile

    y_train = data['Creditability']
    X_train = data.drop('Creditability', axis=1)
    del data

    # 逻辑回归
    n_splits = 5
    cv = StratifiedKFold(n_splits, shuffle=True)
    estimator = LogisticRegression(penalty='l2', C=1, solver='liblinear')
    logistic_score_report(estimator, X_train, y_train, cv, name='estimator')

    # lasso逻辑回归
    lassocv = LogisticRegressionCV(penalty='l1', solver='liblinear', max_iter=1000)
    lassocv.fit(X_train, y_train)
    # optimal C: the inverse of lambda
    print(lassocv.C_)
    # The coefficients
    print('Lasso Coefficients: \n', lassocv.coef_)
    #
    estimator = LogisticRegression(penalty='l2', C=lassocv.C_[0], solver='liblinear')
    logistic_score_report(estimator, X_train, y_train, cv, name='estimator')


def aus():
    path = r'D:\ppd_project\raw_data\test_data_01\aus.csv'
    data = pd.read_csv(path)
    # profile = data.profile_report(title='German Data')
    # profile.to_file(output_file="Training_Data.html")
    # del profile

    y_train = data['y']
    X_train = data.drop('y', axis=1)
    del data

    # 逻辑回归
    n_splits = 5
    cv = StratifiedKFold(n_splits, shuffle=True)
    estimator = LogisticRegression(penalty='l2', C=1, solver='liblinear')
    logistic_score_report(estimator, X_train, y_train, cv, name='estimator')

    # lasso逻辑回归
    lassocv = LogisticRegressionCV(penalty='l1', solver='liblinear', max_iter=1000)
    lassocv.fit(X_train, y_train)
    # optimal C: the inverse of lambda
    print(lassocv.C_)
    # The coefficients
    print('Lasso Coefficients: \n', lassocv.coef_)
    #
    estimator = LogisticRegression(penalty='l2', C=lassocv.C_[0], solver='liblinear')
    logistic_score_report(estimator, X_train, y_train, cv, name='estimator')


if __name__ == '__main__':
    german()
    # aus()
    pass
