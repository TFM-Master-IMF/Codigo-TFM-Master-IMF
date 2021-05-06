import pandas as pd
import numpy as np
import requests
from YahooFinanceHistory import YahooFinanceHistory
from bs4 import BeautifulSoup
from gdeltdoc import GdeltDoc, Filters
from pytrends.request import TrendReq
from os.path import dirname, abspath

import pandas_datareader as pdr
import datetime

def fill_missing_dates(df):
    df = df[~df.index.duplicated()]
    idx = pd.date_range(df.index[0].strftime("%d-%m-%y"), df.index[-1].strftime("%d-%m-%y"))
    df = df.reindex(idx)
    df.fillna(method='ffill', inplace=True)
    return df

def extract_data_from_fred():
    start_fred = datetime.datetime(2018, 1, 1)
    end_fred = datetime.datetime(2019, 12, 31)

    # todos los indicadores no estan seasonalmente ajustados
    data_from_fred = pdr.DataReader(['T10YIE',
                                     'DGS10',
                                     'T10Y2Y',
                                     'TEDRATE',
                                     'DTWEXBGS',
                                     'VIXCLS',
                                     'DCOILWTICO',
                                     'DEXUSUK',
                                     'DEXCAUS',
                                     'USEPUINDXD',
                                     'ICERATES1200EUR5Y',
                                     'BAMLHYH0A0HYM2TRIV',
                                     'BAMLCC0A0CMTRIV',
                                     'WILL5000INDFC'],
                                    'fred',
                                    start_fred,
                                    end_fred)
    data_from_fred = data_from_fred.fillna(method='backfill')
    data_from_fred[['TEDRATE', 'ICERATES1200EUR5Y']] = data_from_fred[['TEDRATE', 'ICERATES1200EUR5Y']].fillna(method='ffill')
    data_from_fred = data_from_fred.rename(columns={'T10YIE': '10-Year Breakeven Inflation Rate',
                                                    'DGS10': '10-Year Treasury Constant Maturity Rate',
                                                    'T10Y2Y': '10-Year Treasury Constant Maturity Minus 2-Year Treasury Constant Maturity',
                                                    'TEDRATE': 'TED Spread',
                                                    'DTWEXBGS': 'Trade Weighted U.S. Dollar Index: Broad, Goods and Services',
                                                    'VIXCLS': 'CBOE Volatility Index',
                                                    'DCOILWTICO': 'Crude Oil Prices: West Texas Intermediate (WTI)',
                                                    'DEXUSUK': 'U.S. / U.K. Foreign Exchange Rate',
                                                    'DEXCAUS': 'Canada / U.S. Foreign Exchange Rate',
                                                    'USEPUINDXD': 'Economic Policy Uncertainty Index for United States',
                                                    'ICERATES1200EUR5Y': 'ICE Swap Rates, 12:00 P.M. (London Time), Based on Euros, 5 Year Tenor',
                                                    'BAMLHYH0A0HYM2TRIV': 'ICE BofA US High Yield Index Total Return Index Value',
                                                    'BAMLCC0A0CMTRIV': ' ICE BofA US Corporate Index Total Return Index Value',
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


def extract_data_from_yahoo_finance(description, symbol, days_back):
    if symbol == 'BTC-USD':
        df = YahooFinanceHistory(symbol, days_back).get_quote().loc[:, ['Date', 'Volume', 'Adj Close']]
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index(['Date'], inplace=True)
        df.rename(columns={'Adj Close': description, 'Volume': 'Bitcoin_Volume'}, inplace=True)
        df = fill_missing_dates(df)
        return df
    else:
        df = YahooFinanceHistory(symbol, days_back).get_quote().loc[:, ['Date', 'Adj Close']]
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index(['Date'], inplace=True)
        df.rename(columns={'Adj Close': description}, inplace=True)
        df = fill_missing_dates(df)
        return df


def extract_data_from_gdelt(word, s_date, e_date):
    f = Filters(
        keyword=word,
        start_date=s_date,
        end_date=e_date
    )
    gd = GdeltDoc()
    df = gd.timeline_search("timelinevol", f)
    df = df.set_index('datetime')
    df.index = df.index.date
    df.index.name = 'Date'
    df.rename(columns={'Volume Intensity': 'Gdelt_Volume_Intensity'}, inplace=True)
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
        df.rename(columns={'Bitcoin': 'Google_Trends'}, inplace=True)
        df = fill_missing_dates(df)
    return df


def make_data_stationary(dataset):
    frames = []
    for col in dataset.columns:
        series_log = np.log(dataset[col])
        series_log_dif = (series_log - series_log.shift(1))*100
        series_log_dif.dropna(inplace=True)
        frames.append(series_log_dif)
    return pd.concat(frames, axis=1, join='inner')


def main():
    # Getting data related to Bitcoin from Google Trends and Gdelt
    frames = [extract_data_from_google_trends('Bitcoin', '2017-01-01 2021-01-01'),
              extract_data_from_gdelt('Bitcoin', '2017-01-01', '2021-01-01')]

    # Getting data from interactive graphs using multiple URLs
    with open('URLs.txt', 'r') as f:
        elements = [line.rstrip('\n').split(',') for line in f]
    for url, content in elements:
        frames.append(extract_data_from_interactive_chart(url, content))

    # Getting data from Yahoo Finance
    with open('Yahoo_Finance.txt', 'r') as f:
        elements = [line.strip().rstrip('\n').split(',') for line in f]
    for description, symbol in elements:
        frames.append(extract_data_from_yahoo_finance(description, symbol, 365*5))

    frames.append(extract_data_from_fred())
    # Merging the distinct dataframes and making the data stationary
    # database = make_data_stationary(pd.concat(frames))
    database = pd.concat(frames, axis=1, join='inner')
    database = database[('2018-01-01' <= database.index) & (database.index < '2020-01-01')]
    database.to_csv(dirname(dirname(abspath(__file__))) + '\Ficheros Outputs\Datos.csv',
                    index_label='Date')
    print(frames)
    print(database)


if __name__ == "__main__":
    main()
