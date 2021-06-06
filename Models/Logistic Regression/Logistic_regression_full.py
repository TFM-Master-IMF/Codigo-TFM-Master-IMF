from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

from os.path import dirname, abspath
import sys

sys.path.insert(0, dirname(dirname(abspath(__file__))))

from Hyperparameter_optimization import evaluate_hyperparameter
from utils import read_data, split_train_test, plot_roc_curve, read_log_dataset, read_raw_dataset


def main(load_dataset):
    np.random.seed(0)

    pipeline = Pipeline([
        ("scaler", MinMaxScaler()),
        ("classifier", LogisticRegression())
    ])

    dataset = load_dataset()

    X_train, X_test, y_train, y_test = split_train_test(dataset)

    #params = evaluate_hyperparameter(pipeline, X_train, X_val, y_train, y_val)

    #model = pipeline["classifier"].set_params(**params)
    model = pipeline['classifier'].set_params()
    #model.fit(np.row_stack([X_train, X_val]), np.concatenate([y_train, y_val]))
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    print('\nAccuracy achieved with the test set: ', accuracy_score(y_test, y_pred))
    print('Precision achieved with the test set: ', precision_score(y_test, y_pred))
    print('Recall achieved with the test set: ', recall_score(y_test, y_pred))
    print('F1 Score achieved with the test set: ', f1_score(y_test, y_pred))
    #plot_roc_curve(y_test, y_pred_proba)


if __name__ == "__main__":
    main(read_data)
    main(read_log_dataset)
    main(read_raw_dataset)
