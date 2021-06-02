from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

from os.path import dirname, abspath
import sys
sys.path.insert(0, dirname(dirname(abspath(__file__))))

from Hyperparameter_optimization import evaluate_hyperparameter
from utils import read_log_dataset, split_train_test, plot_roc_curve


def main():
    np.random.seed(0)

    pipeline = Pipeline([
        ("scaler", MinMaxScaler()),
        ("classifier", LogisticRegression())
    ])

    #solo logs
    dataset_log = read_log_dataset()

    X_train_log, X_val_log, X_test_log, y_train_log, y_val_log, y_test_log = split_train_test(dataset_log)

    params_log = evaluate_hyperparameter(pipeline, X_train_log, X_val_log, y_train_log, y_val_log)

    model_log = pipeline["classifier"].set_params(**params_log)
    model_log.fit(np.row_stack([X_train_log, X_val_log]), np.concatenate([y_train_log, y_val_log]))
    y_pred_log = model_log.predict(X_test_log)
    y_pred_proba_log = model_log.predict_proba(X_test_log)[:, 1]

    print('\nAccuracy achieved with the test set: ', accuracy_score(y_test_log, y_pred_log))
    print('Precision achieved with the test set: ', precision_score(y_test_log, y_pred_log))
    print('Recall achieved with the test set: ', round(recall_score(y_test_log, y_pred_log), 2))
    print('F1 Score achieved with the test set: ', round(f1_score(y_test_log, y_pred_log), 2))
    plot_roc_curve(y_test_log, y_pred_proba_log)

if __name__ == "__main__":
    main()
