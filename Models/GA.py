import numpy as np
import pandas as pd
import random
import matplotlib.pyplot
from os.path import dirname, abspath
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from utils import read_data, split_train_test, plot_roc_curve

def main():
    np.random.seed(0)
    
    dataset = pd.read_csv(dirname(dirname(abspath(__file__))) + '/Ficheros Outputs/Datos.csv', sep=';', decimal=',')
    dataset.set_index('Date', inplace=True)

    X_train, X_val, X_test, y_train, y_val, y_test = split_train_test(dataset)


if __name__ == "__main__":
    main()