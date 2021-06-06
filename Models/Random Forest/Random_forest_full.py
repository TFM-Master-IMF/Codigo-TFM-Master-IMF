from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from sklearn.ensemble import BaggingClassifier

from os.path import dirname, abspath
import sys
sys.path.insert(0, dirname(dirname(abspath(__file__))))

from Hyperparameter_optimization import evaluate_hyperparameter
from utils import read_data, split_train_test, plot_roc_curve


def main():
    np.random.seed(0)
    
    dataset = read_data()

    X_train, X_test, y_train, y_test = split_train_test(dataset)

    #params = evaluate_hyperparameter(RandomForestClassifier(), X_train, X_val, y_train, y_val)

    #model = RandomForestClassifier(**params)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    #model.fit(np.row_stack([X_train, X_val]), np.concatenate([y_train, y_val]))
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    print('\nAccuracy achieved with the test set: ', accuracy_score(y_test, y_pred))
    print('Precision achieved with the test set: ', precision_score(y_test, y_pred))
    print('Recall achieved with the test set: ', recall_score(y_test, y_pred))
    print('F1 Score achieved with the test set: ', f1_score(y_test, y_pred))
    #plot_roc_curve(y_test, y_pred_proba)

    model = BaggingClassifier(base_estimator=RandomForestClassifier(**params),
                       n_estimators=10, random_state=0)

    model.fit(np.row_stack([X_train, X_val]), np.concatenate([y_train, y_val]))
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    print('\nAccuracy achieved with the test set: ', accuracy_score(y_test, y_pred))
    print('Precision achieved with the test set: ', precision_score(y_test, y_pred))
    print('Recall achieved with the test set: ', round(recall_score(y_test, y_pred), 2))
    print('F1 Score achieved with the test set: ', round(f1_score(y_test, y_pred), 2))
    plot_roc_curve(y_test, y_pred_proba)

if __name__ == "__main__":
    main()
