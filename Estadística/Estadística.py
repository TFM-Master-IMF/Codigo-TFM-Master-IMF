import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from os.path import dirname, abspath
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import pacf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf


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
        stationarity_adf, adf = adf_test(dataset.loc[:, col])
        stationarity_kpss, kpss = kpss_test(dataset.loc[:, col])
        adf_results.append(adf)
        kpss_results.append(kpss)
        stationarity_results.append((col, stationarity_adf, stationarity_kpss))
    df_adf = pd.DataFrame(adf_results)
    df_adf.set_index('Variable', inplace=True)
    df_kpss = pd.DataFrame(kpss_results)
    df_kpss.set_index('Variable', inplace=True)
    df_adf.to_csv(dirname(dirname(abspath(__file__))) + '\Ficheros Outputs\ADFLag'
                  + str(lag) + '.csv')
    df_kpss.to_csv(dirname(dirname(abspath(__file__))) + '\Ficheros Outputs\KPSSLag'
                   + str(lag) + '.csv')
    return stationarity_results


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


def check_autocorrelations(dataset):
    print('Autocorrelations of ACF test:')
    best_lags = []
    for col in dataset.columns:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            acf_abs_result = [abs(ele) for ele in list(acf(dataset[col], nlags=10))[1:]]
            index_of_max_acf = acf_abs_result.index(max(acf_abs_result)) + 1
            best_lags.append((col, index_of_max_acf))
            plot_acf(dataset[col], lags=60)
            plt.show()
    return best_lags


def check_correlations(dataset):
    local_dataset = dataset.copy()
    local_dataset_log = logarithmic_transformation(local_dataset)
    corr_matrix = pd.DataFrame(local_dataset_log.corr(method='pearson'))
    corr_matrix['Bitcoin_USD'].sort_values(ascending=False).to_csv(dirname(dirname(abspath(__file__)))
                                                                   + '\Ficheros Outputs\Correlations.csv', index=True)
    print(corr_matrix['Bitcoin_USD'].sort_values(ascending=False))
    """fig, ax = plt.subplots()
    sns.heatmap(
        round(corr_matrix, 2),
        vmin=-1, vmax=1, center=0,
        cmap=sns.diverging_palette(20, 220, n=200),
        square=True,
        annot=True,
        xticklabels=True,
        yticklabels=True
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    )
    plt.show()"""


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


def main():
    dataset = pd.read_csv(dirname(dirname(abspath(__file__))) + '\Ficheros Outputs\Datos.csv',
                          index_col='Date', parse_dates=['Date'])
    check_correlations(dataset)
    dataset_log_dif, lag = make_data_stationary(dataset)
    print(dataset_log_dif, lag)
    # best_lags = check_autocorrelations(dataset_log)
    d = dirname(dirname(abspath(__file__)))
    print(d)


if __name__ == "__main__":
    main()
