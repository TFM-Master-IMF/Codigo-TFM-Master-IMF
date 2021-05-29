import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import os
from arch.unitroot import ADF, PhillipsPerron
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.stattools import kpss
from os.path import dirname, abspath


def set_bitcoin_sign_change(dataset, lag):
    local_dataset = dataset.copy()
    local_dataset['Bitcoin sign change'] = local_dataset['Bitcoin Stock Price (USD)'].shift(lag) - local_dataset[
        'Bitcoin Stock Price (USD)']
    local_dataset.dropna(axis=0, inplace=True)
    local_dataset['Bitcoin sign change'] = local_dataset['Bitcoin sign change'].apply(lambda row: 0 if row < 0 else 1)
    return local_dataset['Bitcoin sign change']


def adf_test(timeseries):
    """Function that determines if a time series is stationary by performing the Augmented Dickey-Fuller test."""

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        adftest = ADF(timeseries.replace([np.inf, -np.inf], np.nan).dropna())
        results = {'Variable': timeseries.name, 'Test Statistic': round(adftest.stat, 2),
                   'P-value': round(adftest.pvalue, 2), '1%': round(adftest.critical_values['1%'], 2),
                   '5%': round(adftest.critical_values['5%'], 2), '10%': round(adftest.critical_values['10%'], 2)}
    if adftest.pvalue > 0.05:
        results['Type of time series'] = 'Non stationary'
        return False, results
    else:
        results['Type of time series'] = 'Stationary'
        return True, results


def pp_test(timeseries):
    """Function that determines if a time series is stationary by performing the Phillips-Perron test."""
    pptest = PhillipsPerron(timeseries.replace([np.inf, -np.inf], np.nan).dropna())
    results = {'Variable': timeseries.name, 'Test Statistic': round(pptest.stat, 2),
               'P-value': round(pptest.pvalue, 2), '1%': round(pptest.critical_values['1%'], 2),
               '5%': round(pptest.critical_values['5%'], 2), '10%': round(pptest.critical_values['10%'], 2)}
    if pptest.pvalue > 0.05:
        results['Type of time series'] = 'Non stationary'
        return False, results
    else:
        results['Type of time series'] = 'Stationary'
        return True, results


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
    pp_results = []
    # kpss_results = []
    for col in dataset.columns:
        stationarity_adf, adf_data = adf_test(dataset.loc[:, col])
        stationarity_pp, pp_data = pp_test(dataset.loc[:, col])
        # stationarity_kpss, kpss_data = kpss_test(dataset.loc[:, col])
        adf_results.append(adf_data)
        pp_results.append(pp_data)
        # kpss_results.append(kpss_data)
        stationarity_results.append((col, stationarity_adf, stationarity_pp))
    df_adf = pd.DataFrame(adf_results)
    df_adf.set_index('Variable', inplace=True)
    df_pp = pd.DataFrame(pp_results)
    df_pp.set_index('Variable', inplace=True)
    # df_kpss = pd.DataFrame(kpss_results)
    # df_kpss.set_index('Variable', inplace=True)
    df_adf.to_csv(dirname(dirname(abspath(__file__))) + '/Ficheros Outputs/Stationarity-Tests/ADFLag'
                  + str(lag) + '.csv')
    df_pp.to_csv(dirname(dirname(abspath(__file__))) + '/Ficheros Outputs/Stationarity-Tests/PPLag'
                 + str(lag) + '.csv')
    # df_kpss.to_csv(dirname(dirname(abspath(__file__))) + '/Ficheros Outputs/Stationarity-Tests/KPSSLag'
    #               + str(lag) + '.csv')
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
        local_dataset[col] = (local_dataset[col] - local_dataset[col].shift(lag))
        local_dataset.rename(columns={col: 'Log Return ' + col}, inplace=True)
    local_dataset.dropna(inplace=True)
    return local_dataset


def make_data_stationary(dataset):
    """Function that computes the logarithmic and differencing transformation of the input dataset variables until
    the data becomes stationary."""

    lag = 1
    local_dataset = dataset.copy()
    dataset_log = logarithmic_transformation(local_dataset)
    dataset_log_dif = dataset_log.copy()
    dataset_log_dif = differencing_transformation(dataset_log, lag)
    while any(False in ele for ele in check_stationarity(dataset_log_dif, lag)):    
        lag += 1
        dataset_log_dif = differencing_transformation(dataset_log, lag)       
    return dataset_log_dif


def grangers_causality_test(dataset, variables, max_lag, target='Log Return Bitcoin Stock Price (USD)', test='ssr_chi2test'):
    """Check Granger Causality of all the Time series.
    The target is the response variable, the independent variables are the predictors."""

    causality = {}
    for var in variables:
        if var != target:
            test_result = grangercausalitytests(dataset[[target, var]], maxlag=max_lag, verbose=False)
            p_values = np.array([[round(test_result[i + 1][0][test][1], 4), int(i + 1)] for i in range(max_lag)])
            df = pd.DataFrame(p_values, columns=['P-Value', 'Lag'])
            df.to_csv(dirname(dirname(abspath(__file__))) + '/Ficheros Outputs/Causality-Tests/'
                      + var + ' - ' + target + '.csv')
            if all([i > 0.01 for i in p_values[:, 0]]):
                causality[var] = False
            else:
                causality[var] = True
    return causality


def crosscorr(datax, datay, lag=0):
    """Lag-N cross correlation"""
    local_datay = datay.copy()
    return datax.corr(local_datay.shift(lag), method='pearson')


def cross_correlation(dataset, target):
    cross_correlations = pd.DataFrame(np.zeros((31, len(dataset.columns) - 1)),
                                      columns=dataset.columns[dataset.columns != target])
    a = 8
    b = 2
    c = 1
    fig = plt.figure(figsize=(15, 30))
    for col in dataset.columns:
        if col != target:
            cross_correlations[col] = [crosscorr(dataset[target], dataset[col], i) for i in range(0, 31)]
            plt.subplot(a, b, c)
            plt.title('Log Return Bitcoin Stock Price (USD) - ' + col)
            plt.xlabel('Time lags in number of days')
            plt.ylabel('Cross-Correlation value')
            plt.plot([i for i in range(0, 31)], cross_correlations[col])
            plt.grid()
            c += 1
    plt.tight_layout()
    fig.savefig(dirname(dirname(abspath(__file__)))
                + '/Ficheros Outputs/Cross-Correlations/' + 'Cross-Correlations' + '.png')
    best_lags = {}
    abs_cross_correlations = cross_correlations.abs()
    lags = abs_cross_correlations.idxmax()
    i = 0
    for col in cross_correlations.columns:
        best_lags[col] = (col + ' ' + 'lag' + ' ' + str(lags[i]), cross_correlations.loc[lags[i], col], lags[i])
        i += 1
    return best_lags


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
    if not os.path.exists(dirname(dirname(abspath(__file__)))
                          + '/Ficheros Outputs/Stationarity-Tests'):
        os.makedirs(dirname(dirname(abspath(__file__)))
                    + '/Ficheros Outputs/Stationarity-Tests')
    if not os.path.exists(dirname(dirname(abspath(__file__)))
                          + '/Ficheros Outputs/Causality-Tests'):
        os.makedirs(dirname(dirname(abspath(__file__)))
                    + '/Ficheros Outputs/Causality-Tests')

    # Reading data
    dataset = pd.read_csv(dirname(dirname(abspath(__file__))) + '/Ficheros Outputs/Datos.csv',
                          index_col='Date', parse_dates=['Date'], sep=';', decimal=',')
    bitcoin_sign_change = set_bitcoin_sign_change(dataset.loc['2018-01-01':'2020-01-01'], -1)
    dataset = dataset.loc['2018-01-01':'2019-12-31']

    # Making all the variables of the dataset stationary
    dataset_log_dif = make_data_stationary(dataset)  

    # Checking for causality
    grangers_causality_test_results = grangers_causality_test(dataset_log_dif, dataset_log_dif.columns, 30)
    selected_variables = [key for key, value in grangers_causality_test_results.items() if value]
    selected_variables.append('Log Return Bitcoin Stock Price (USD)')

    # Selection of the best lags for each independent variables based on their correlations with the Bitcoin Price
    cross_correlation_results = cross_correlation(dataset_log_dif.loc[:, selected_variables],
                                                  'Log Return Bitcoin Stock Price (USD)')
    
    # Generation of the final dataset
    frames = []
    for key, value in cross_correlation_results.items():
        feature = dataset_log_dif[key].shift(value[2]).rename(value[0], inplace=True)
        frames.append(dataset[key[11:]]) # Getting de variables of the initial dataset (we need to slice the Log Return tag)
        frames.append(feature)
    # frames.append(dataset_log_dif['Log Return Bitcoin Stock Price (USD)']) 
    # frames.append('Bitcoin Stock Price (USD)')
    frames.append(bitcoin_sign_change)
    final_dataset = pd.concat(frames, axis=1, join='inner')
    final_dataset.dropna(axis=0, inplace=True)
    final_dataset.to_csv(dirname(dirname(abspath(__file__))) + '/Ficheros Outputs/DatosFinales.csv',
                         index_label='Date', sep=';', decimal='.')

    print(final_dataset)


if __name__ == "__main__":
    main()
