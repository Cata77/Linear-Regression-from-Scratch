import numpy as np


class CustomLinearRegression:

    def __init__(self, *, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.coefficient = None
        self.intercept = None

    def fit(self, X: list[float], y: list[float]):
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if not isinstance(y, np.ndarray):
            y = np.array(y)

        if self.fit_intercept:
            X = np.column_stack((np.ones(len(X)), X))

        XtX_inv = np.linalg.inv(X.T.dot(X))
        self.coefficient = XtX_inv.dot(X.T).dot(y)

        if self.fit_intercept:
            self.intercept = self.coefficient[0]
            self.coefficient = self.coefficient[1:]


if __name__ == '__main__':
    x = [4.0, 4.5, 5, 5.5, 6.0, 6.5, 7.0]
    y = [33, 42, 45, 51, 53, 61, 62]
    regression = CustomLinearRegression()
    regression.fit(x, y)

    output_dict = {'Intercept': regression.intercept, 'Coefficient': regression.coefficient}
    print(output_dict)
