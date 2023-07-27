import math

import numpy as np


class CustomLinearRegression:

    def __init__(self, *, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.coefficient = None
        self.intercept = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        if self.fit_intercept:
            X = np.column_stack((np.ones(len(X)), X))

        XtX_inv = np.linalg.inv(X.T.dot(X))
        self.coefficient = XtX_inv.dot(X.T).dot(y)

        if self.fit_intercept:
            self.intercept = self.coefficient[0]
            self.coefficient = self.coefficient[1:]

    def predict(self, X):
        if self.fit_intercept:
            X = np.column_stack((np.ones(len(X)), X))

        if self.intercept is not None:
            self.coefficient = np.insert(self.coefficient, 0, self.intercept)

        return X.dot(self.coefficient)

    def r2_score(self, y, y_hat):
        y_mean = np.mean(y)
        return 1 - np.sum((y - y_hat) ** 2) / np.sum((y - y_mean) ** 2)

    def rmse(self, y, y_hat):
        MSE = 1 / len(y) * np.sum((y - y_hat) ** 2)
        return math.sqrt(MSE)


if __name__ == '__main__':
    capacity = [0.9, 0.5, 1.75, 2.0, 1.4, 1.5, 3.0, 1.1, 2.6, 1.9]
    age = [11, 11, 9, 8, 7, 7, 6, 5, 5, 4]
    cost_per_ton = [21.95, 27.18, 16.9, 15.37, 16.03, 18.15, 14.22, 18.72, 15.4, 14.69]
    regression = CustomLinearRegression(fit_intercept=True)
    X = np.column_stack((np.array(capacity), np.array(age)))
    regression.fit(X, np.array(cost_per_ton))
    y_pred = regression.predict(X)

    output_dict = {'Intercept': regression.intercept, 'Coefficient': np.array(regression.coefficient[1:]),
                   'R2': regression.r2_score(np.array(cost_per_ton), y_pred),
                   'RMSE': regression.rmse(np.array(cost_per_ton), y_pred)}
    print(output_dict)
