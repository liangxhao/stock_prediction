import os
import pandas as pd

START_DATE = '2004-08-19'
END_DATE = '2021-06-11'

def read_data(file_name: str) -> pd.DataFrame:
    data = pd.read_csv(file_name, index_col="Date", parse_dates=True, na_values=['nan'])
    data = data.drop(columns=["Close"])
    data = data.rename(columns={
        "Adjusted Close": "Close"
    })
    data.sort_index(axis=0, inplace=True)
    data = cutoff_data(data)

    assert data.isnull().values.any() == False
    return data


def cutoff_data(data: pd.DataFrame) -> pd.DataFrame:
    date = pd.date_range(START_DATE, END_DATE, freq='B')
    date = pd.DataFrame(index=date)
    data = data.join(date, how="right")
    data.fillna(method='ffill', inplace=True)

    return data