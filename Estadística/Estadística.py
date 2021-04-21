import numpy as np
import pandas as pd
import warnings
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss


def adf_test(timeseries):
    """ Function that determines a time series seasonality by performing the Augmented Dickey-Fuller test.
    Parameters:
        timeseries: time series on which to run the seasonality test
    """

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        adftest = adfuller(timeseries.replace([np.inf, -np.inf], np.nan).dropna(), autolag='AIC')
    # print('Is {} data stationary?'.format(col_name))
    # print('ADF test statistic = {:.3f}'.format(adftest[0]))
    # print('P-value = {:.3f}'.format(adftest[1]))
    # print('Critical values :')
    # for k, v in adftest[4].items():
    #    if k not in ['1%', '2.5%', '10%']:
    #        print('\t{}: {} -> The data is {} stationary with {}% confidence'
    #              .format(k, v, 'not' if v < adftest[0] else '', 100 - int(k[:-1])))
    if (adftest[1] >= 0.05) & (adftest[0] > adftest[4]['5%']):
        return False
    else:
        return True


def kpss_test(timeseries):
    """Function that determines a time series seasonality by performing the
    Kwiatkowski–Phillips–Schmidt–ShinDickey-Fuller test

    Parameters:
        timeseries: time series on which to run the seasonality test
    """
    with warnings.catch_warnings():
        # Ignoring the warnings generate due to p-values out of table range
        warnings.filterwarnings("ignore")
        kpsstest = kpss(timeseries.replace([np.inf, -np.inf], np.nan).dropna(), regression='c', nlags="auto")
    # print('KPPS test statistic = {:.3f}'.format(kpsstest[0]))
    # print('P-value = {:.3f}'.format(kpsstest[1]))
    # print('Critical values :')
    # for k, v in kpsstest[3].items():
    #    if k not in ['1%', '2.5%', '10%']:
    #        print('\t{}: {} -> The data is {} stationary with {}% confidence'
    #              .format(k, v, 'not' if v < kpsstest[0] else '', 100 - float(k[:-1])))
    if (kpsstest[1] <= 0.05) & (kpsstest[3]['5%'] < kpsstest[0]):
        return False
    else:
        return True


def check_stationarity(dataset):
    stationarity_results = []
    for col in dataset.columns:
        stationarity_results.append((col, adf_test(dataset.loc[:, col]), kpss_test(dataset.loc[:, col])))
    return stationarity_results


def make_data_stationary(dataset):
    frames = []
    for col in dataset.columns:
        series_log = np.log(dataset[col])
        series_log_dif = (series_log - series_log.shift(2))*100
        series_log_dif.dropna(inplace=True)
        frames.append(series_log_dif)
    return pd.concat(frames, axis=1, join='inner')


def main():
    dataset = pd.read_csv('D:\Documentos\Master Big Data & Business Analytics\TFM\Datos.csv', index_col='Date',
                          parse_dates=['Date'])
    print(dataset)
    print(check_stationarity(dataset))
    dataset = make_data_stationary(dataset)
    print(dataset)
    print(check_stationarity(dataset))


if __name__ == "__main__":
    main()
