import pandas as pd
from prediction.supervised import train
from prediction.supervised import wrangle_data
from prediction.supervised import preprocess_data

if __name__ == '__main__':
    # data = pd.read_csv('data/merged_information_clean.csv', encoding='utf-8')
    # wrangle_data.wrangle_data(data)

    # clean_filled = pd.read_csv('data/clean_filled.csv', encoding='utf-8')
    # preprocess_data.preprocess(clean_filled)

    ml_ready = pd.read_csv('data/ml_ready.csv')
    train.train(ml_ready)
