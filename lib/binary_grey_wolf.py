import random
import pandas as pd
import numpy as np
from lib.estimator import KnnError, LogError, RfError
import matplotlib.pyplot as plt


class BinaryGreyWolfOptimizer:

    def __init__(self, X_train, y_train, N, T, estimator, error_name):
        self.N = N
        self.T = T
        self.estimator = estimator
        self.error_name = error_name
        self.X_train = X_train
        self.y_train = y_train
        self.X_alpha_error, self.X_beta_error, self.X_delta_error = None, None, None
        self.X_alpha, self.X_beta, self.X_delta = None, None, None

    def __init_population(self):
        self.D = self.X_train.shape[1]
        self.X = np.zeros(shape=(self.N, self.D))
        for i in range(self.N):
            for d in range(self.D):
                if random.random() > 0.5:
                    self.X[i, d] = 1

    def __init_hierarchy(self):
        error_array = np.zeros(shape=self.N)
        for i in range(self.N):
            X_i = self.X[i, :]
            error = self.estimator.get_error(self.X_train, self.y_train, X_i, self.error_name)
            error_array[i] = error
        idx = np.argsort(error_array)
        self.X_alpha = self.X[idx[0], :].reshape(self.D)
        self.X_beta = self.X[idx[1], :].reshape(self.D)
        self.X_delta = self.X[idx[2], :].reshape(self.D)
        self.X_alpha_error = error_array[idx[0]]
        self.X_beta_error = error_array[idx[1]]
        self.X_delta_error = error_array[idx[2]]

    def polt(self):
        pass

    def bgwo(self):
        self.__init_population()
        self.__init_hierarchy()

        X_alpha, X_beta, X_delta = self.X_alpha, self.X_beta, self.X_delta
        alpha_error, beta_error, delta_error = self.X_alpha_error, self.X_beta_error, self.X_delta_error
        X = self.X

        # 记录每次迭代的结果error
        error_res = []
        for t in range(1, self.T + 1):
            a = 2 - 2 * (t / self.T)
            for i in range(self.N):
                for d in range(self.D):
                    C1, C2, C3 = 2 * random.random(), 2 * random.random(), 2 * random.random()
                    Dalpha = abs(C1 * X_alpha[d] - X[i, d])
                    Dbeta = abs(C2 * X_beta[d] - X[i, d])
                    Ddelta = abs(C3 * X_delta[d] - X[i, d])
                    A1, A2, A3 = 2 * a * random.random() - a, 2 * a * random.random() - a, 2 * a * random.random() - a

                    X1 = X_alpha[d] - A1 * Dalpha
                    X2 = X_beta[d] - A2 * Dbeta
                    X3 = X_delta[d] - A3 * Ddelta
                    Xn = (X1 + X2 + X3) / 3
                    TF = 1 / (np.exp(-10 * (Xn - 0.5)))
                    if TF >= random.random():
                        X[i, d] = 1
                    else:
                        X[i, d] = 0

            # 更新头目
            for i in range(self.N):
                error = self.estimator.get_error(self.X_train, self.y_train, X[i, :], self.error_name)
                if error < alpha_error:
                    alpha_error = error
                    X_alpha = X[i, :]
                if alpha_error < error < beta_error:
                    beta_error = error
                    X_beta = X[i, :]
                if alpha_error < beta_error < error < delta_error:
                    delta_error = error
                    X_delta = X[i, :]
            print(alpha_error, t)
            error_res.append(abs(alpha_error))

        # self.plot_iter(self.error_name, error_res)
        return error_res, X_alpha

    def ibgwo(self):
        self.__init_population()
        self.__init_hierarchy()

        X_alpha, X_beta, X_delta = self.X_alpha, self.X_beta, self.X_delta
        alpha_error, beta_error, delta_error = self.X_alpha_error, self.X_beta_error, self.X_delta_error
        X = self.X

        # 记录每次迭代的结果error
        error_res = []
        for t in range(1, self.T + 1):
            a = 2 - 2 * np.sin(np.pi / 2 * t / self.T)
            w = abs(alpha_error + beta_error + delta_error)
            w1 = 1 - abs(alpha_error) / w
            w2 = 1 - abs(beta_error) / w
            w3 = 1 - abs(delta_error) / w
            for i in range(self.N):
                for d in range(self.D):
                    C1, C2, C3 = 2 * random.random(), 2 * random.random(), 2 * random.random()
                    Dalpha = abs(C1 * X_alpha[d] - X[i, d])
                    Dbeta = abs(C2 * X_beta[d] - X[i, d])
                    Ddelta = abs(C3 * X_delta[d] - X[i, d])
                    A1, A2, A3 = 2 * a * random.random() - a, 2 * a * random.random() - a, 2 * a * random.random() - a

                    X1 = X_alpha[d] - A1 * Dalpha
                    X2 = X_beta[d] - A2 * Dbeta
                    X3 = X_delta[d] - A3 * Ddelta

                    Xn = X1 * w1 + X2 * w2 + X3 * w3
                    TF = 1 / (np.exp(-10 * (Xn - 0.5)))
                    if TF >= random.random():
                        X[i, d] = 1
                    else:
                        X[i, d] = 0

            # 更新头目
            for i in range(self.N):
                error = self.estimator.get_error(self.X_train, self.y_train, X[i, :], self.error_name)
                if error < alpha_error:
                    alpha_error = error
                    X_alpha = X[i, :]
                if alpha_error < error < beta_error:
                    beta_error = error
                    X_beta = X[i, :]
                if alpha_error < beta_error < error < delta_error:
                    delta_error = error
                    X_delta = X[i, :]
            print(alpha_error, t)
            error_res.append(abs(alpha_error))

        # self.plot_iter(self.error_name, error_res)
        return error_res, X_alpha

    @staticmethod
    def plot_iter(error_name, error_res):
        x_lab = [i + 1 for i in range(len(error_res))]
        plt.plot(x_lab, error_res)
        plt.title('the relationship between' + error_name + 'and iterations')
        plt.ylabel(error_name)
        plt.xlabel("num of iterations")
        plt.show()


def repeat_iter(X_train, y_train, simulation, N, T, estimator, mode, error_name):
    res_error = pd.DataFrame()
    res_X_alpha = pd.DataFrame()

    for s in range(simulation):
        bwo = BinaryGreyWolfOptimizer(X_train, y_train, N=N, T=T, estimator=estimator, error_name=error_name)
        if mode == 'bwo':
            error_ls, X_feature_index = bwo.bgwo()
            res_error = res_error.append(error_ls)
            res_X_alpha = res_X_alpha.append(X_feature_index)
        else:
            error_ls, X_feature_index = bwo.ibgwo()
            res_error = res_error.append(error_ls)
            res_X_alpha = res_X_alpha.append(X_feature_index)

    print(res_error)
    print(res_X_alpha)
    pass


def german(path, estimator_name, error_name, mode, simulation=10, N=10, T=30):
    data = pd.read_csv(path)
    if 'aus' in path:
        y_train = data['y']
        X_train = data.drop('y', axis=1)
    else:
        y_train = data['Creditability']
        X_train = data.drop('Creditability', axis=1)
    del data

    if estimator_name == 'KNN':
        error_estimator = KnnError(5, 5)
    else:
        error_estimator = RfError(5, 5)

    repeat_iter(X_train, y_train, simulation, N, T, error_estimator, mode, error_name)


if __name__ == '__main__':
    path = r'raw_data/test_data_01/german_credit.csv'
    german(path, 'KNN', 'accuracy', mode='bwo')
    german(path, 'KNN', 'accuracy', mode='ibwo')
    german(path, 'RF', 'accuracy', mode='bwo')
    german(path, 'RF', 'accuracy', mode='ibwo')

    path = r'raw_data/test_data_01/aus.csv'
    german(path, 'KNN', 'accuracy', mode='bwo')
    german(path, 'KNN', 'accuracy', mode='ibwo')
    german(path, 'RF', 'accuracy', mode='bwo')
    german(path, 'RF', 'accuracy', mode='ibwo')
