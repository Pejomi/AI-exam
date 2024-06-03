import pandas as pd
import wrangle_data
import preprocess_data
import train


if __name__ == '__main__':
    # data = pd.read_csv('data/merged_information_clean.csv', encoding='utf-8', nrows=1000)
    # wrangle_data.wrangle_data(data)

    # cleaned_data = pd.read_csv('data/cleaned_data.csv', encoding='utf-8')
    # preprocess_data.preprocess(cleaned_data)

    train.train_decision_tree()
