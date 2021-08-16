device = "cuda"
test_days = 60
batch_size = 128
n_steps = 64
# columns = ['Volume', 'Close']
columns = ['Close']
model_path = 'models_lstm'
symbol = "GOOG"

# START_DATE = '2012-01-06'
# END_DATE = '2019-12-31'

START_DATE = '2013-01-06'
END_DATE = '2018-12-31'

DATASETS_PATH = r"F:\workspace\datasets\stock_market_data\forbes2000\csv"
