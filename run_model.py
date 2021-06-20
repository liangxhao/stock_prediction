import data_prepare

apple = data_prepare.read_data("AAPL")
apple = data_prepare.cutoff_data(apple)

print(apple)