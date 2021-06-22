import os
import pandas as pd

START_DATE = '2012-01-01'
END_DATE = '2016-12-31'
DATASETS_PATH = r"/home/liangxhao/workspace/datasets/stock_market_data/forbes2000/csv"

def read_data(symbol: str) -> pd.DataFrame:
    file_name = os.path.join(DATASETS_PATH, symbol + ".csv")
    data = pd.read_csv(file_name, index_col="Date", parse_dates=True, na_values=['nan'])
    data = data.drop(columns=["Close"])
    data = data.rename(columns={
        "Adjusted Close": "Close"
    })
    data.sort_index(axis=0, inplace=True)

    return data


def cutoff_data(company: pd.DataFrame) -> pd.DataFrame:
    date = pd.date_range(START_DATE, END_DATE, freq='B')
    date = pd.DataFrame(index=date)
    company = company.join(date, how="right")
    company.fillna(method='ffill', inplace=True)

    return company