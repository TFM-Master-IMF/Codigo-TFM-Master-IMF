import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import os
# import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from os.path import dirname, abspath
from statsmodels.tsa.stattools import grangercausalitytests


def adf_test(timeseries):
    """Function that determines if a time series is stationary by performing the Augmented Dickey-Fuller test."""

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        adftest = adfuller(timeseries.replace([np.inf, -np.inf], np.nan).dropna(), autolag='AIC')


        resultados = {'Variable': timeseries.name, 'Estadísticas de prueba': round(adftest[0], 2),
                      'P-valor': round(adftest[1], 2), '1%': round(adftest[4]['1%'], 2),
                      '5%': round(adftest[4]['5%'], 2), '10%': round(adftest[4]['10%'], 2)}
    if (adftest[1] >= 0.05) & (adftest[0] > adftest[4]['5%']):
        resultados['Tipo de serie'] = 'No estacionaria'
        return False, resultados
    else:
        resultados['Tipo de serie'] = 'Estacionaria'
        return True, resultados


def kpss_test(timeseries):
    """Function that determines if a time series is stationary by performing the
     Kwiatkowski–Phillips–Schmidt–ShinDickey-Fuller test"""

    with warnings.catch_warnings():
        # Ignoring the warnings generated due to p-values out of table range
        warnings.filterwarnings("ignore")
        kpsstest = kpss(timeseries.replace([np.inf, -np.inf], np.nan).dropna(), regression='c', nlags="auto")
        resultados = {'Variable': timeseries.name, 'Estadísticas de prueba': round(kpsstest[0], 2),
                      'P-valor': round(kpsstest[1], 2), '1%': round(kpsstest[3]['1%'], 2),
                      '5%': round(kpsstest[3]['5%'], 2), '10%': round(kpsstest[3]['10%'], 2)}
    if (kpsstest[1] <= 0.05) & (kpsstest[3]['5%'] < kpsstest[0]):
        resultados['Tipo de serie'] = 'No estacionaria'
        return False, resultados
    else:
        resultados['Tipo de serie'] = 'Estacionaria'
        return True, resultados


def check_stationarity(dataset, lag):
    """Function that checks the stationarity of the time series variables present in a dataset applying the ADF and
    KPPS tests. Moreover, it stores the results of both test in two different csv files."""

    stationarity_results = []
    adf_results = []
    kpss_results = []
    for col in dataset.columns:
        stationarity_adf, adf_data = adf_test(dataset.loc[:, col])
        stationarity_kpss, kpss_data = kpss_test(dataset.loc[:, col])
        adf_results.append(adf_data)
        kpss_results.append(kpss_data)
        stationarity_results.append((col, stationarity_adf, stationarity_kpss))
    df_adf = pd.DataFrame(adf_results)
    df_adf.set_index('Variable', inplace=True)
    df_kpss = pd.DataFrame(kpss_results)
    df_kpss.set_index('Variable', inplace=True)
    df_adf.to_csv(dirname(dirname(abspath(__file__))) + '/Ficheros Outputs/ADFLag'
                  + str(lag) + '.csv')
    df_kpss.to_csv(dirname(dirname(abspath(__file__))) + '/Ficheros Outputs/KPSSLag'
                   + str(lag) + '.csv')
    return stationarity_results


def logarithmic_transformation(dataset):
    """Function that applies the logarithmic transformation to all the time series variables present in the dataset"""

    local_dataset = dataset.copy()
    for col in local_dataset.columns:
        local_dataset[col] = np.log(local_dataset[col])
    return local_dataset


def differencing_transformation(dataset, lag):
    """Function that applies the differencing method to the variables of a dataset given a certain lag value"""

    local_dataset = dataset.copy()
    for col in local_dataset.columns:
        local_dataset[col] = (local_dataset[col] - local_dataset[col].shift(lag)) * 100
    local_dataset.dropna(inplace=True)
    return local_dataset


def make_data_stationary(dataset):
    """Function that computes the logarithmic and differencing transformation of the input dataset variables until
    the data becomes stationary."""

    lag = 0
    local_dataset = dataset.copy()
    dataset_log = logarithmic_transformation(local_dataset)
    dataset_log_dif = dataset_log.copy()
    while any(False in ele for ele in check_stationarity(dataset_log_dif, lag)):
        lag += 1
        dataset_log_dif = differencing_transformation(dataset_log, lag)
    return dataset_log_dif, lag


def crosscorr(datax, datay, lag=0):
    """ Lag-N cross correlation"""

    return datax.corr(datay.shift(lag), method='pearson')


def cross_correlation(dataset, target):
    cross_correlations = pd.DataFrame(np.zeros((301 - (-300), len(dataset.columns) - 1)),
                                      columns=dataset.columns[dataset.columns != target],
                                      index=["lag_" + str(i) for i in range(-300, 301)])
    for col in dataset.columns:
        if col != target:
            cross_correlations[col] = [crosscorr(dataset[target], dataset[col], i) for i in range(-300, 301)]
            fig, ax = plt.subplots()
            ax.plot([i for i in range(-300, 301)], cross_correlations[col])
            ax.set_title('Bitcoin-' + col)
            plt.xlabel('Time lags, in number of days (n)')
            plt.ylabel('Cross-Correlation value')
            plt.grid()
            fig.savefig(dirname(dirname(abspath(__file__)))
                        + '/Ficheros Outputs/Cross-Correlations/' + col + '.png')

    return cross_correlations


def grangers_causation_matrix(dataset, variables, max_lag, test='ssr_chi2test', verbose=False):
    """Check Granger Causality of all possible combinations of the Time series.
    The rows are the response variable, columns are predictors."""

    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(dataset[[r, c]], maxlag=max_lag, verbose=False)
            p_values = [round(test_result[i + 1][0][test][1], 4) for i in range(max_lag)]
            if verbose:
                print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df


def main():
    # Creation of needed directories
    if not os.path.exists(dirname(dirname(abspath(__file__)))
                          + '/Ficheros Outputs'):
        os.makedirs(dirname(dirname(abspath(__file__)))
                    + '/Ficheros Outputs')
    if not os.path.exists(dirname(dirname(abspath(__file__)))
                          + '/Ficheros Outputs/Cross-Correlations'):
        os.makedirs(dirname(dirname(abspath(__file__)))
                    + '/Ficheros Outputs/Cross-Correlations')
    # Reading dataframe with the required data
    dataset = pd.read_csv(dirname(dirname(abspath(__file__))) + '/Ficheros Outputs/Datos.csv',
                          index_col='Date', parse_dates=['Date'])
    dataset_log_dif, lag = make_data_stationary(dataset)
    # print(dataset_log_dif, lag)
    cross_correlation(dataset_log_dif, "Bitcoin_USD")
    # grangers_causality_matrix = grangers_causation_matrix(dataset_log_dif, dataset_log_dif.columns, 7)
    # print(grangers_causality_matrix.loc['Bitcoin_USD_y', :].map(lambda p_value: p_value <= 0.05))


if __name__ == "__main__":
    main()
