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
        if self.intercept is not None:
            self.coefficient.insert(self.intercept, 0)

        return X.dot(self.coefficient)


if __name__ == '__main__':
    x = [4, 4.5, 5, 5.5, 6, 6.5, 7]
    w = [1, -3, 2, 5, 0, 3, 6]
    z = [11, 15, 12, 9, 18, 13, 16]
    y = [33, 42, 45, 51, 53, 61, 62]

    regression = CustomLinearRegression(fit_intercept=False)
    X = np.column_stack((np.array(x), np.array(w), np.array(z)))
    regression.fit(X, np.array(y))
    y_pred = regression.predict(X)
    print(y_pred)