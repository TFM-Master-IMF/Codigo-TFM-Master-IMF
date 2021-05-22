import pandas as pd
from os.path import dirname, abspath
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt


def read_data():
    data = pd.read_csv(dirname(dirname(abspath(__file__))) + '/Ficheros Outputs/DatosFinales.csv', sep=';', decimal=',')
    data.set_index('Date', inplace=True)
    print(data.head())
    return data


def split_train_test(data):
    X = data.drop("Bitcoin sign change", axis=1)
    y = data["Bitcoin sign change"]

    # División en train y test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # División del set de entrenamiento en entrenamiento y validación
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=False)

    return X_train, X_val, X_test, y_train, y_val, y_test


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


if __name__ == "__main__":
    data = read_data()
