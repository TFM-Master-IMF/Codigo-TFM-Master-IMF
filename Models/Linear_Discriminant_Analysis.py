from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np

from Hyperparameter_optimization import evaluate_hyperparameter
from utils import read_data, split_train_test, plot_roc_curve


def main():
    dataset = read_data()

    X_train, X_val, X_test, y_train, y_val, y_test = split_train_test(dataset)

    params = evaluate_hyperparameter(LinearDiscriminantAnalysis(), X_train, X_val, y_train, y_val)

    model = LinearDiscriminantAnalysis(**params)
    model.fit(np.row_stack([X_train, X_val]), np.concatenate([y_train, y_val]))
    y_pred = model.predict_proba(X_test)[:, 1]

    plot_roc_curve(y_test, y_pred)


if __name__ == "__main__":
    main()
