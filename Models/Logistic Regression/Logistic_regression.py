from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import pandas as pd
import os

from os.path import dirname, abspath
import sys

sys.path.insert(0, dirname(dirname(abspath(__file__))))

from Hyperparameter_optimization import evaluate_hyperparameter
from utils import read_data, split_train_test, plot_roc_curve, read_log_dataset, read_raw_dataset, get_features_importance, plot_features_importance, load_model


def build_model_exogenous_variables(dataset):
    #print(dataset)
    X_train, X_test, y_train, y_test = split_train_test(dataset)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = round(accuracy_score(y_test, y_pred), 4)
    precision = round(precision_score(y_test, y_pred), 4)
    recall = round(recall_score(y_test, y_pred), 4)
    f1_score_ = round(f1_score(y_test, y_pred), 4)

    return accuracy, precision, recall, f1_score_ 


def main(load_dataset):
    np.random.seed(0)

    pipeline = Pipeline([
        ("scaler", MinMaxScaler()),
        ("classifier", LogisticRegression())
    ])

    dataset = load_dataset()

    X_train, X_test, y_train, y_test = split_train_test(dataset)

    #params = evaluate_hyperparameter(pipeline, X_train, X_test, y_train, y_test)
    
    #model = pipeline["classifier"].set_params(**params)
    model = pipeline['classifier'].set_params()
    # model.fit(np.row_stack([X_train, X_val]), np.concatenate([y_train, y_val]))
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    #y_pred_proba = model.predict_proba(X_test)[:, 1]

    print('\nAccuracy achieved with the test set: ', round(accuracy_score(y_test, y_pred), 4))
    print('Precision achieved with the test set: ', round(precision_score(y_test, y_pred), 4))
    print('Recall achieved with the test set: ', round(recall_score(y_test, y_pred), 4))
    print('F1 Score achieved with the test set: ', round(f1_score(y_test, y_pred), 4))
    # plot_roc_curve(y_test, y_pred_proba)

    random_forest = load_model('RandomForestClassifier_results.pkl')
    ordered_features_importance  = get_features_importance(dataset, random_forest)['VARIABLE'].values
    variables = ['Bitcoin sign change']
    results = []
    
    for feature in ordered_features_importance:
        variables.insert(0, feature)
        """print('\n')
        print(variables)"""
        results.append(build_model_exogenous_variables(dataset.loc[:, variables]))
    
    final_results = pd.DataFrame(results, columns=['Accuracy', 'Precision', 'Recall', 'F1 Score'])
    final_results.index = np.arange(1, len(final_results) + 1)
    final_results.index.name = 'Exogenous Variables'
    print(final_results)
    if not os.path.exists(dirname(dirname(abspath(__file__)))
                          + '/Models Results'):
        os.makedirs(dirname(dirname(abspath(__file__)))
                    + '/Models Results')
    final_results.to_csv(dirname(dirname(abspath(__file__)))+ '/Models Results/Results_Logistic_Regression.csv')    


if __name__ == "__main__":
    #main(read_data)
    main(read_log_dataset)
    #main(read_raw_dataset)
