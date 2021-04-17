import pandas as pd
import requests
import pandas_datareader as pdr
import datetime
from bs4 import BeautifulSoup
from gdeltdoc import GdeltDoc, Filters
from pytrends.request import TrendReq


def fill_missing_dates(df):
    df = df[~df.index.duplicated()]
    idx = pd.date_range(df.index[0].strftime("%d-%m-%y"), df.index[-1].strftime("%d-%m-%y"))
    df = df.reindex(idx)
    df.fillna(method='ffill', inplace=True)
    return df


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
        values.append(value)

    df = pd.DataFrame(list(zip(dates, values)), columns=["Date", content])
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index(['Date'], inplace=True)
    return df


def extract_data_from_yahoo_finance(description, symbol):
    start = datetime.datetime(2011, 1, 1)
    end = datetime.datetime(2020, 12, 31)
    if symbol == 'BTC-USD':
        df = pdr.get_data_yahoo(symbol, start=start, end=end).loc[:, ['Volume', 'Adj Close']]
        df.rename(columns={'Adj Close': description, 'Volume': 'Bitcoin_Volume'}, inplace=True)
        df = fill_missing_dates(df)
        return df
    else:
        df = pdr.get_data_yahoo(symbol, start=start, end=end).loc[:, 'Adj Close'].to_frame()
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


def main():
    # Getting data related to Bitcoin from Google Trends and Gdelt
    frames = [extract_data_from_google_trends('Bitcoin', '2011-01-01 2020-12-31'),
              extract_data_from_gdelt('Bitcoin', '2017-01-01', '2020-12-31')]

    # Getting data from interactive graphs using multiple URLs
    with open('URLs.txt', 'r') as f:
        elements = [line.rstrip('\n').split(',') for line in f]
    for url, content in elements:
        frames.append(extract_data_from_interactive_chart(url, content))

    # Getting data from Yahoo Finance
    with open('Yahoo_Finance.txt', 'r') as f:
        elements = [line.strip().rstrip('\n').split(',') for line in f]
    for description, symbol in elements:
        frames.append(extract_data_from_yahoo_finance(description, symbol))

    database = pd.concat(frames, axis=1, join='inner')
    print(database)
    database.to_csv('D:\Documentos\Master Big Data & Business Analytics\TFM\Datos.csv', index_label='Date')


if __name__ == "__main__":
    main()
