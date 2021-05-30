from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

from Hyperparameter_optimization import evaluate_hyperparameter
from utils import read_data, split_train_test, plot_roc_curve


def main():
    np.random.seed(0)
    
    dataset = read_data()

    X_train, X_val, X_test, y_train, y_val, y_test = split_train_test(dataset)

    params = evaluate_hyperparameter(RandomForestClassifier(), X_train, X_val, y_train, y_val)

    model = RandomForestClassifier(**params)
    model.fit(np.row_stack([X_train, X_val]), np.concatenate([y_train, y_val]))
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    print('Accuracy achieved with the test set: ', accuracy_score(y_test, y_pred))
    plot_roc_curve(y_test, y_pred_proba)


if __name__ == "__main__":
    main()
