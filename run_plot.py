import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import data_vis
import data_prepare


apple = data_prepare.read_data("AAPL")
google = data_prepare.read_data("GOOG")
facebook = data_prepare.read_data("FB")

# vis.vis_pair(apple)
# vis.vis_price(apple)
# vis.vis_line(apple, google, facebook)
# vis.vis_kde(apple)
