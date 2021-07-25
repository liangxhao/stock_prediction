import pandas as pd
import matplotlib.pyplot as plt

def result_plot(preds: pd.DataFrame, trues: pd.DataFrame):
    plt.plot(preds.index, preds['pred'], label='pred')
    plt.plot(trues.index, trues['true'], label='true')
    plt.legend()
    plt.show()


def results_plot(train_result, test_result):
    test_preds, test_trues = test_result
    plt.plot(test_preds.index, test_preds['pred'], label='test')

    show_day = len(test_trues.index) * 10

    train_preds, train_trues = train_result
    plt.plot(train_preds.index[-show_day:], train_preds['pred'][-show_day:], label='train')

    trues = pd.concat([train_trues[-show_day:], test_trues])
    plt.plot(trues.index, trues['true'], label='true')

    plt.legend()
    plt.show()
