import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import data_vis
import data_prepare


apple = data_prepare.read_data("AAPL")
google = data_prepare.read_data("GOOG")
facebook = data_prepare.read_data("FB")

apple = data_prepare.cutoff_data(apple)
google = data_prepare.cutoff_data(google)
facebook = data_prepare.cutoff_data(facebook)


# data_vis.vis_pair(apple)
# data_vis.vis_price(apple)
# data_vis.vis_line(apple, google, facebook)
# data_vis.vis_kde(apple)

p=apple.corr()
p=1