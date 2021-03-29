# Importación de módulos
import datetime
import os
import pandas as pd


def dateparse(time_in_secs):
    """Función que permite transformar las fechas del formato timestamp a date/time """
    return datetime.datetime.fromtimestamp(float(time_in_secs))


def lectura_inicial_datos():
    """Función que permite realizar la lectura inicial de la base de datos de Bitcoin"""

    # Captura del directorio actual
    dirname = os.getcwd()

    # Concatenación del directorio actual con el subdirectorio deseado
    data_path = os.path.join(dirname, r'./Lectura_datos/Base de datos Bitcoin/bitstampUSD.csv')

    # Lectura del csv con los datos iniciales
    return pd.read_csv(data_path, header=0, parse_dates=[0],
                       date_parser=dateparse, index_col=[0])
