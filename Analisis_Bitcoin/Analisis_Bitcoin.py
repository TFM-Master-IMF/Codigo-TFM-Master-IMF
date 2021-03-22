# Librerías dedicadas al procesamiento de los datos
import numpy as np
import pandas as pd

# Librerías para la visualización de los datos
import plotly.graph_objs as go
from plotly.offline import iplot

# Test de Dickey-Fuller aumentado
from statsmodels.tsa.stattools import adfuller


def visual_stationarity_test(title,
                             inicial_range=('2010-04-01', '2021-03-19'), original_timeseries=None,
                             timeseries_without_tendency=None, my_window=None):
    """Función que permite comprobar la estacionalidad de una serie temporal mediante la elaboración de un gráfico
    interactivo."""
    trace_original_timeseries, trace_timeseries_without_tendency, trace_rolmean, trace_rolstd = None, None, None,None

    if original_timeseries is not None:
        trace_original_timeseries = go.Scatter(
            x=original_timeseries.index,
            y=original_timeseries.astype(float),
            name=title,
            opacity=1)

    if timeseries_without_tendency is not None:
        trace_timeseries_without_tendency = go.Scatter(
            x=timeseries_without_tendency.index,
            y=timeseries_without_tendency.astype(float),
            name=title,
            opacity=1)

    if my_window is not None:
        rolmean = pd.Series(original_timeseries).rolling(window=my_window).mean()
        rolstd = pd.Series(original_timeseries).rolling(window=my_window).std()

        trace_rolmean = go.Scatter(
            x=rolmean.index,
            y=rolmean.astype(float),
            name="Media",
            opacity=1)

        trace_rolstd = go.Scatter(
            x=rolstd.index,
            y=rolstd.astype(float),
            name="Desviación estándar",
            opacity=0.8)

    if original_timeseries is not None:
        trace_original_timeseries = go.Scatter(
            x=original_timeseries.index,
            y=original_timeseries.astype(float),
            name='Precio original de cierre del Bitcoin',
            opacity=1)

    traces = [trace_original_timeseries, trace_timeseries_without_tendency, trace_rolmean, trace_rolstd]
    data = [elem for elem in traces if elem is not None]

    layout = dict(
        title='Test visual de la estacionalidad de la variable \'close\'',
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1,
                         label='1m',
                         step='month',
                         stepmode='backward'),
                    dict(count=6,
                         label='6m',
                         step='month',
                         stepmode='backward'),
                    dict(count=12,
                         label='1y',
                         step='month',
                         stepmode='backward'),
                    dict(count=36,
                         label='3y',
                         step='month',
                         stepmode='backward'),
                    dict(step='all')
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            type='date'
        )
    )

    fig = dict(data=data, layout=layout)
    fig['layout']['xaxis'].update(range=inicial_range)
    iplot(fig, filename="Test visual de la estacionalidad de la variable \'close\'")


def ADF_test(timeseries, dataDesc):
    """Función que determina la estacionalidad de una serie temporal mediante la realización del test de
    Dickey-Fuller aumentado. """

    print('Is the {} stationary?'.format(dataDesc))
    dftest = adfuller(timeseries.replace([np.inf, -np.inf], np.nan).dropna(), autolag='AIC')
    print('Test statistic = {:.3f}'.format(dftest[0]))
    print('P-value = {:.3f}'.format(dftest[1]))
    print('Critical values :')
    for k, v in dftest[4].items():
        print('\t{}: {} - The data is {} stationary with {}% confidence'
              .format(k, v, 'not' if v < dftest[0] else '', 100 - int(k[:-1])))
