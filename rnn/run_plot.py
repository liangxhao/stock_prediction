import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rnn.data import data_vis, data_prepare

apple = data_prepare.read_data("AAPL")
google = data_prepare.read_data("GOOG")
amazon = data_prepare.read_data("AMZN")


# data_vis.vis_pair(apple)
# data_vis.vis_price(apple)
# data_vis.vis_line(apple, google, amazon)
data_vis.vis_kde(apple)

