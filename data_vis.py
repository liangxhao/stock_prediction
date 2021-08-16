import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def vis_line(apple, google, facebook):
    # 绘图
    column = "Close"
    # column = "Volume"
    plot_data = pd.concat([apple[column], google[column], facebook[column]], axis=1)
    plot_data.set_axis(["apple", "google", "facebook"], axis='columns', inplace=True)

    plot_data.plot()
    plt.xlabel("Date")
    plt.ylabel(column)
    plt.title(f"{column} Price")
    plt.show()


def vis_price(company):
    plot_data = company[['Open', 'Close']]
    plot_data.plot()
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title(f"Open & Close Price")
    plt.show()


def vis_kde(company):
    sns.kdeplot(x=company['Close'],
                y=company['Volume'],
                clip=[[5, 60], [0, 0.6e9]],
                cmap='Greens',
                shade=True,
                shade_lowest=False
                )
    plt.xlabel("Close Price")
    plt.ylabel("Volume")
    plt.title("KDE for Close Price and Volume")
    plt.show()


def vis_pair(company):
    sns.pairplot(company)
    plt.show()