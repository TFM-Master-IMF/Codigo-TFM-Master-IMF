# Librerías dedicadas al procesamiento de los datos
import numpy as np
import pandas as pd

# Test de Dickey-Fuller aumentado
from statsmodels.tsa.stattools import adfuller
# Test KPSS
from statsmodels.tsa.stattools import kpss


def ADF_test(timeseries, col_name):
    """Función que determina la estacionalidad de una serie temporal mediante la realización del test de
    Dickey-Fuller aumentado. """

    print('Is {} data stationary?'.format(col_name))
    adftest = adfuller(timeseries.replace([np.inf, -np.inf], np.nan).dropna(), autolag='AIC')
    print('ADF test statistic = {:.3f}'.format(adftest[0]))
    print('P-value = {:.3f}'.format(adftest[1]))
    print('Critical values :')
    for k, v in adftest[4].items():
        print('\t{}: {} - The data is {} stationary with {}% confidence'
              .format(k, v, 'not' if v < adftest[0] else '', 100 - int(k[:-1])))


def KPSS_test(timeseries, col_name):
    print('Is {} data stationary?'.format(col_name))
    kpsstest = kpss(timeseries, regression='c', nlags="auto")
    print('KPPS test statistic = {:.3f}'.format(kpsstest[0]))
    print('P-value = {:.3f}'.format(kpsstest[1]))
    print('Critical values :')
    for k, v in kpsstest[3].items():
        if k != '2.5%':
            print('\t{}: {} - The data is {} stationary with {}% confidence'
                  .format(k, v, 'not' if v < kpsstest[0] else '', 100 - float(k[:-1])))


datos = pd.read_csv('D:\Documentos\Master Big Data & Business Analytics\TFM\Datos.csv', index_col='Date',
                    parse_dates=['Date'])
for col in datos.columns:
    # ADF_test(datos.loc[:, col], col)
    KPSS_test(datos.loc[:, col], col)
