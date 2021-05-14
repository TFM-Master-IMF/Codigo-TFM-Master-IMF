import requests
import pandas as pd
from io import StringIO


class YahooFinanceHistory:
    quote_link = 'https://query1.finance.yahoo.com/v7/finance/download/{quote}?period1={dfrom}&period2={' \
                 'dto}&interval=1d&events=history&includeAdjustedClose=true '

    def __init__(self, symbol, datefrom, dateto):
        self.symbol = symbol
        self.session = requests.Session()
        self.datefrom = datefrom
        self.dateto = dateto

    def get_quote(self):
        url = self.quote_link.format(quote=self.symbol, dfrom=int(self.datefrom.timestamp()),
                                     dto=int(self.dateto.timestamp()))
        response = self.session.get(url)
        response.raise_for_status()
        return pd.read_csv(StringIO(response.text), parse_dates=['Date'])
