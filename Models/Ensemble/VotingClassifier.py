from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from Models.utils import split_train_test, read_data, read_log_dataset, read_raw_dataset, plot_features_importance


def main(load_dataset):
    dataset = load_dataset()

    X_train, X_test, y_train, y_test = split_train_test(dataset)

    model = VotingClassifier(estimators=[
        ('rf', RandomForestClassifier()),
        # ('lda', LinearDiscriminantAnalysis()),
        # ('lr', LogisticRegression()),
        # ('mlp', MLPClassifier()),
        ('qda', QuadraticDiscriminantAnalysis()),
        # ('svc', SVC()),
        # ('xgb', XGBClassifier()),
    ],
                             voting='hard')

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # y_pred_proba = model.predict_proba(X_test)[:, 1]
    print('\nAccuracy achieved with the test set: ', accuracy_score(y_test, y_pred))
    print('Precision achieved with the test set: ', precision_score(y_test, y_pred))
    print('Recall achieved with the test set: ', round(recall_score(y_test, y_pred), 2))
    print('F1 Score achieved with the test set: ', round(f1_score(y_test, y_pred), 2))

    # plot_features_importance(dataset, model)


if __name__ == "__main__":
    # main(read_data)
    main(read_log_dataset)
    # main(read_raw_dataset)
