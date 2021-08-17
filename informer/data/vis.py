import pandas as pd
import matplotlib.pyplot as plt

SHOW_DETAIL = True

def result_plot(preds: pd.DataFrame, trues: pd.DataFrame):
    plt.plot(preds.index, preds['pred'], label='pred')
    plt.plot(trues.index, trues['true'], label='true')
    plt.legend()
    plt.show()


def results_plot(train_result, test_result):
    test_preds, test_trues = test_result

    #  10, 2
    if SHOW_DETAIL:
        show_day = len(test_trues.index) * 2
    else:
        show_day = 0

    train_preds, train_trues = train_result
    trues = pd.concat([train_trues[-show_day:], test_trues])

    plt.plot(trues.index, trues['true'], label='true')
    plt.plot(test_preds.index, test_preds['pred'], label='test')
    plt.plot(train_preds.index[-show_day:], train_preds['pred'][-show_day:], label='train')

    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.title('Close Price Fitting')
    plt.show()
