from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

from os.path import dirname, abspath
import sys
sys.path.insert(0, dirname(dirname(abspath(__file__))))

from Hyperparameter_optimization import evaluate_hyperparameter
from utils import read_log_dataset, split_train_test, plot_roc_curve


def main():
    np.random.seed(0)
    
    dataset_log = read_log_dataset()

    X_train_log, X_test_log, y_train_log, y_test_log = split_train_test(dataset_log)

    #params_log = evaluate_hyperparameter(XGBClassifier(use_label_encoder=False, eval_metric='logloss'), X_train_log, X_val_log, y_train_log, y_val_log)

    #model = XGBClassifier(**params)
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train_log, y_train_log)
    #model.fit(np.row_stack([X_train, X_val]), np.concatenate([y_train, y_val]))

    y_pred_log = model.predict(X_test_log)
    y_pred_proba_log = model.predict_proba(X_test_log)[:, 1]

    print('\nAccuracy achieved with the test set: ', accuracy_score(y_test_log, y_pred_log))
    print('Precision achieved with the test set: ', precision_score(y_test_log, y_pred_log))
    print('Recall achieved with the test set: ', recall_score(y_test_log, y_pred_log))
    print('F1 Score achieved with the test set: ', f1_score(y_test_log, y_pred_log))
    #plot_roc_curve(y_test_log, y_pred_proba_log)


if __name__ == "__main__":
    main()
