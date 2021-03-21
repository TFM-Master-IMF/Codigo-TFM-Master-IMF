import requests
import pandas as pd
from datetime import date
from datetime import datetime
from datetime import timedelta


def string_data_to_timestamp(string):
    """Función que permite pasar una cadena de caracteres que contenga una fecha a un timestamp"""

    element = datetime.strptime(string, "%d/%m/%Y %H:%M")
    return datetime.timestamp(element)


def data_to_dataframe(data):
    """Función que transforma los datos de entrada (data) de tipo JSON a un dataframe de pandas"""

    df = pd.DataFrame.from_dict(data)
    df['Timestamp'] = pd.to_datetime(df['time'], unit='s')
    df.drop(['time', 'conversionType', 'conversionSymbol'], axis=1, inplace=True)
    df.set_index('Timestamp', inplace=True)
    return df


def fetch(baseurl, parameters):
    """Función que realiza las peticiones de los datos a una api de criptomonedas, selecciona los datos deseados y
    los devuelve como un dataframe de pandas """

    response = requests.get(baseurl, params=parameters)
    json_response = response.json()['Data']['Data']
    return data_to_dataframe(json_response)


def get_historical_bitcoin_data(from_sym='', to_sym='', timeframe='', limit=2000):
    """Función que obtiene los datos históricos del precio del Bitcoin. Al comienzo de la misma se establecen los
    parametros necesarios para relizar las peticiones correctas a la api. Debido a que la api no devuelve más de 2000
    registros, se ha utilizado un bucle para poder obtener todos los datos historicos de la criptomoneda deseada """

    list_df_bitcoin_data = []
    baseurl = 'https://min-api.cryptocompare.com/data/v2/histo'
    baseurl += timeframe
    parameters = {'fsym': from_sym,  # Símbolo de la criptomoneda de interés
                  'tsym': to_sym,  # Símbolo de moneda a la que convertir las variables relacionadas con el precio
                  'limit': limit,  # Número máximo de instancias devueltas por la api
                  'toTs': string_data_to_timestamp(date.today().strftime('%d/%m/%Y %H:%M')),
                  # Marca de tiempo que establece los datos a devolver por la api
                  'api_key': '5c2058afa3861e6026e6ec4f776ed42be6a3629b13fd653a6986bd76cac6a658'
                  }

    while string_data_to_timestamp('22/05/2010 00:00') < parameters['toTs']:
        try:
            list_df_bitcoin_data.append(fetch(baseurl, parameters))
            parameters['toTs'] = datetime.timestamp((list_df_bitcoin_data[-1].index[0] - timedelta(
                hours=1)).to_pydatetime())  # Restamos una hora a la variable flag para evitar la repetición de instancias al hacer las peticiones a la api
        except:
            parameters['toTs'] = datetime.timestamp((list_df_bitcoin_data[-1].index[0] - timedelta(
                hours=1)).to_pydatetime())  # Restamos una hora a la variable flag para evitar la repetición de instancias al hacer las peticiones a la api

    return list_df_bitcoin_data[::-1]


data = pd.concat(get_historical_bitcoin_data('BTC', 'USD', 'hour'))
data.to_csv(path_or_buf='Base de datos Bitcoin/Bitcoinhourly.csv', index=True, index_label='Timestamp')
