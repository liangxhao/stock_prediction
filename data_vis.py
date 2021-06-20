import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

start_date = '2019-01-01'
end_date = '2020-04-30'

def vis_line(apple, google, facebook):
    # 绘图
    date = pd.date_range(start_date, end_date, freq='B')
    column = "Close"

    date = pd.DataFrame(index=date)
    apple = apple.join(date, how="inner")
    google = google.join(date, how="inner")
    facebook = facebook.join(date, how="inner")

    plot_data = pd.concat([apple[column], google[column], facebook[column]], axis=1)
    plot_data.set_axis(["apple", "google", "facebook"], axis='columns', inplace=True)

    plot_data.plot()
    plt.xlabel("Date")
    plt.ylabel(column)
    plt.title(f"{column} Price")
    plt.show()


def vis_price(company):
    date = pd.date_range(start_date, end_date, freq='B')
    date = pd.DataFrame(index=date)
    company = company.join(date, how="inner")

    plot_data = company[['Open', 'Close']]
    plot_data.plot()
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title(f"Open & Close Price")
    plt.show()

def vis_kde(company):
    sns.kdeplot(x=company['Close'],
                y=company['Volume'],
                clip=[[-40, 40], [0, 0.6e9]],
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