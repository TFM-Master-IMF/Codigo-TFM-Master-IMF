import pandas as pd
import numpy as np
import requests
import os
import pandas_datareader as pdr
import datetime
from YahooFinanceHistory import YahooFinanceHistory
from bs4 import BeautifulSoup
from gdeltdoc import GdeltDoc, Filters
from pytrends.request import TrendReq
from os.path import dirname, abspath


def fill_missing_dates(df):
    df = df[~df.index.duplicated()]
    idx = pd.date_range(df.index[0].strftime("%d-%m-%y"), df.index[-1].strftime("%d-%m-%y"))
    df = df.reindex(idx)
    df.fillna(method='ffill', inplace=True)
    return df


def extract_data_from_fred(start_date, end_date):
    # todos los indicadores no estan seasonalmente ajustados
    data_from_fred = pdr.DataReader(['T10YIE',
                                     'DGS10',
                                     'TEDRATE',
                                     'DTWEXBGS',
                                     'VIXCLS',
                                     'DCOILWTICO',
                                     'DEXUSUK',
                                     'DEXCAUS',
                                     'USEPUINDXD',
                                     'WILL5000INDFC'],
                                    'fred',
                                    start_date,
                                    end_date)
    data_from_fred = data_from_fred.fillna(method='backfill')
    data_from_fred[['TEDRATE']] = data_from_fred[['TEDRATE']].fillna(method='ffill')
    data_from_fred = data_from_fred.rename(columns={'T10YIE': '10-Year Breakeven Inflation Rate',
                                                    'DGS10': '10-Year Treasury Constant Maturity Rate',
                                                    'TEDRATE': 'TED Spread',
                                                    'DTWEXBGS': 'Trade Weighted U.S. Dollar Index: Broad, Goods and '
                                                                'Services',
                                                    'VIXCLS': 'CBOE Volatility Index',
                                                    'DCOILWTICO': 'Crude Oil Prices: West Texas Intermediate (WTI)',
                                                    'DEXUSUK': 'U.S. _ U.K. Foreign Exchange Rate',
                                                    'DEXCAUS': 'Canada _ U.S. Foreign Exchange Rate',
                                                    'USEPUINDXD': 'Economic Policy Uncertainty Index for United States',
                                                    'WILL5000INDFC': 'Wilshire 5000 Total Market Full Cap Index'
                                                    })
    data_from_fred.index = data_from_fred.index.rename('Date')
    return data_from_fred


def extract_data_from_interactive_chart(url, content):
    session = requests.Session()
    page = session.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    data = str(soup.find_all('script')[4])
    data = data.split('d = new Dygraph(document.getElementById("container"),')[1].split(', {labels: ')[0]
    dates = []
    values = []

    for i in range(data.count('new Date')):
        date = data.split('new Date("')[i + 1].split('"')[0]
        value = data.split('"),')[i + 1].split(']')[0]
        dates.append(date)
        if value != 'null':
            values.append(float(value))
        else:
            values.append(None)

    df = pd.DataFrame(list(zip(dates, values)), columns=["Date", content])
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index(['Date'], inplace=True)
    df = fill_missing_dates(df)
    return df


def extract_data_from_yahoo_finance(description, symbol, start_date, end_date):
    if symbol == 'BTC-USD':
        df = YahooFinanceHistory(symbol, start_date, end_date).get_quote().loc[:, ['Date', 'Volume', 'Adj Close']]
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index(['Date'], inplace=True)
        df.rename(columns={'Adj Close': description, 'Volume': 'Bitcoin Volume'}, inplace=True)
        df = fill_missing_dates(df)
        return df
    else:
        df = YahooFinanceHistory(symbol, start_date, end_date).get_quote().loc[:, ['Date', 'Adj Close']]
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index(['Date'], inplace=True)
        df.rename(columns={'Adj Close': description}, inplace=True)
        df = fill_missing_dates(df)
        return df


def extract_data_from_gdelt(word, start_date, end_date):
    f = Filters(
        keyword=word,
        start_date=start_date,
        end_date=end_date
    )
    gd = GdeltDoc()
    df = gd.timeline_search("timelinevol", f)
    df = df.set_index('datetime')
    df.index = df.index.date
    df.index.name = 'Date'
    df.rename(columns={'Volume Intensity': 'Gdelt Volume Intensity'}, inplace=True)
    df = fill_missing_dates(df)
    return df


def extract_data_from_google_trends(keywords, dates):
    pytrend = TrendReq()
    pytrend.build_payload(
        kw_list=[keywords],
        cat=0,
        geo='',
        timeframe=dates
    )
    df = pytrend.interest_over_time()
    if not df.empty:
        df = df.drop(labels=['isPartial'], axis='columns')
        df.index.name = 'Date'
        df.rename(columns={'Bitcoin': 'Google Trends'}, inplace=True)
        df = fill_missing_dates(df)
    return df


def make_data_stationary(dataset):
    frames = []
    for col in dataset.columns:
        series_log = np.log(dataset[col])
        series_log_dif = (series_log - series_log.shift(1)) * 100
        series_log_dif.dropna(inplace=True)
        frames.append(series_log_dif)
    return pd.concat(frames, axis=1, join='inner')


def main():
    # Creation of needed directories
    if not os.path.exists(dirname(dirname(abspath(__file__)))
                          + '/Ficheros Outputs'):
        os.makedirs(dirname(dirname(abspath(__file__)))
                    + '/Ficheros Outputs')

    # Getting data related to Bitcoin from Google Trends and Gdelt
    frames = [extract_data_from_google_trends('Bitcoin', '2017-01-01 2021-01-31'),
              extract_data_from_gdelt('Bitcoin', '2017-01-01', '2021-01-31')]

    # Getting data from interactive graphs using multiple URLs
    with open('URLs.txt', 'r') as f:
        elements = [line.rstrip('\n').split(',') for line in f]
    for url, content in elements:
        frames.append(extract_data_from_interactive_chart(url, content))

    # Getting data from Fred
    frames.append(extract_data_from_fred(datetime.datetime(2017, 1, 1), datetime.datetime(2021, 1, 31)))

    # Getting data from Yahoo Finance
    with open('Yahoo_Finance.txt', 'r') as f:
        elements = [line.strip().rstrip('\n').split(',') for line in f]
    for description, symbol in elements:
        frames.append(extract_data_from_yahoo_finance(description, symbol,
                                                      datetime.datetime(2017, 1, 1), datetime.datetime(2021, 1, 31)))

    # Merging the distinct dataframes
    database = pd.concat(frames, axis=1, join='inner')
    database = database[('2018-01-01' <= database.index) & (database.index < '2021-01-01')]
    database.to_csv(dirname(dirname(abspath(__file__))) + '/Ficheros Outputs/Datos.csv',
                    index_label='Date')

    print(database)


if __name__ == "__main__":
    main()
