import os

device = "cuda"
test_days = 96
batch_size = 128
n_steps = 64

# columns = ['Volume', 'Close']
columns = ['Close']
model_path = os.path.join(os.path.dirname(__file__), 'checkpoints_lstm')

symbol = "GOOG"


START_DATE = '2013-01-06'
END_DATE = '2018-12-31'

DATASETS_PATH = r"E:\workspace\datasets\stock_market_data\forbes2000\csv"


# AAPL: 1980-12-12, 2021-08-16
# GOOG: 2004-08-19, 2021-08-16
# AMZN: 1997-05-15, 2021-08-16
