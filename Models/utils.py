import pandas as pd
from os.path import dirname, abspath
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import pickle
import os


def read_data():
    data = pd.read_csv(dirname(dirname(abspath(__file__))) + '/Ficheros Outputs/DatosFinales.csv', sep=';', decimal='.')
    data.set_index('Date', inplace=True)
    # data.drop('Bitcoin Stock Price (USD)', axis=1, inplace=True)
    print(data.head())
    return data


def read_log_dataset():
    data = pd.read_csv(dirname(dirname(abspath(__file__))) + '/Ficheros Outputs/DatosFinales.csv', sep=';', decimal='.')
    data.set_index('Date', inplace=True)
    dataset_log = data.loc[:,
                  data.columns.str.contains('^log', case=False) | data.columns.str.contains('^Bitcoin sign change',
                                                                                            case=False)]
    print(dataset_log.head())
    return dataset_log


def read_raw_dataset():
    data = pd.read_csv(dirname(dirname(abspath(__file__))) + '/Ficheros Outputs/DatosFinales.csv', sep=';', decimal='.')
    data.set_index('Date', inplace=True)
    dataset_raw = data.loc[:, ~data.columns.str.contains('^log', case=False)]
    print(dataset_raw.head())
    return dataset_raw


def split_train_test(data):
    X = data.drop("Bitcoin sign change", axis=1)
    y = data["Bitcoin sign change"]

    # División en train y test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # División del set de entrenamiento en entrenamiento y validación
    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=False)

    return X_train, X_test, y_train, y_test  # X_val,y_val


def plot_roc_curve(y_val, y_pred):
    auc = roc_auc_score(y_val, y_pred)
    title = f"AUC: {auc:.4f}"
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title(title)
    fpr, tpr, th = roc_curve(y_val, y_pred)
    ax.plot(fpr, tpr)
    ax.set_ylabel("False positive rate [%]")
    ax.set_xlabel("True positive rate [%]")
    ax.plot([0, 1], [0, 1], linestyle="--")
    plt.show()


def get_features_importance(dataset, model):
    X = dataset.drop("Bitcoin sign change", axis=1)

    importance = pd.DataFrame({
        "VARIABLE": X.columns,
        "IMPORTANCE": model.feature_importances_ if hasattr(model, 'feature_importances_') else model.coef_
    }).sort_values("IMPORTANCE", ascending=False)

    return importance


def plot_features_importance(importance):

    print("\n ########## %s variables importance ###########") #% type(model).__name__
    for index, row in importance.iterrows():
        print('Feature: %s, Score: %.5f' % (row["VARIABLE"], row["IMPORTANCE"]))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Feature importances")
    ax.barh(importance["VARIABLE"], importance["IMPORTANCE"])
    plt.show()


def save_model(model):
    classifier = model["classifier"] if type(model) == Pipeline else model

    # save the results
    path = dirname(abspath(__file__))
    directory = '/artifacts'
    result_file_name = '%s_results.pkl' % type(classifier).__name__

    if not os.path.exists(directory):
        os.makedirs(directory)

    pickle.dump(classifier, open(os.path.join(path + directory, result_file_name), 'wb'))


def load_model(name):
    path = dirname(abspath(__file__))
    directory = '/artifacts'

    loaded_model = pickle.load(open(os.path.join(path + directory, name), 'rb'))
    return loaded_model

if __name__ == "__main__":
    data = read_data()
