import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)


def wrangle_data(data):
    print(len(data))

    data_cut = pd.DataFrame(data[
        ['Accident_Severity', 'Speed_limit', 'Weather_Conditions', 'Road_Surface_Conditions', 'Vehicle_Manoeuvre']])

    data_cut.replace('Data missing or out of range', np.nan, inplace=True)
    data_cut.replace('', np.nan, inplace=True)
    data_cut_cleaned = data_cut.dropna(how='any')

    data_cut_cleaned.to_csv('data/cleaned_data.csv', index=False)


def calculate_midpoint(range_str):
    if ' - ' in range_str:
        # It's a range, calculate the midpoint
        lower, upper = map(int, range_str.split(' - '))
        return (lower + upper) / 2
    else:
        # It's a single value (e.g., "80" for "Over 75"), directly convert to int
        return int(range_str)
