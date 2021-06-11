from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

from os.path import dirname, abspath
import sys
sys.path.insert(0, dirname(dirname(abspath(__file__))))

from Hyperparameter_optimization import evaluate_hyperparameter
from utils import read_raw_dataset, split_train_test, plot_roc_curve

def main():
    np.random.seed(0)
    dataset_raw = read_raw_dataset()

    X_train_raw, X_test_raw, y_train_raw, y_test_raw = split_train_test(dataset_raw)

    #params_raw = evaluate_hyperparameter(QuadraticDiscriminantAnalysis(), X_train_raw, X_val_raw, y_train_raw, y_val_raw)

    #model_raw = QuadraticDiscriminantAnalysis(**params_raw)
    model_raw = QuadraticDiscriminantAnalysis()
    model_raw.fit(X_train_raw, y_train_raw)
    #model_raw.fit(np.row_stack([X_train_raw, X_val_raw]), np.concatenate([y_train_raw, y_val_raw]))

    y_pred_raw = model_raw.predict(X_test_raw)
    y_pred_proba_raw = model_raw.predict_proba(X_test_raw)[:, 1]

    print('\nAccuracy achieved with the test set: ', accuracy_score(y_test_raw, y_pred_raw))
    print('Precision achieved with the test set: ', precision_score(y_test_raw, y_pred_raw))
    print('Recall achieved with the test set: ', recall_score(y_test_raw, y_pred_raw))
    print('F1 Score achieved with the test set: ', f1_score(y_test_raw, y_pred_raw))
    #plot_roc_curve(y_test_raw, y_pred_proba_raw)


if __name__ == "__main__":
    main()