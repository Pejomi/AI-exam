import pandas as pd

pd.set_option('display.max_columns', None)


def preprocess(cleaned_data):
    data_df = cleaned_data

    data_df_encoded = pd.get_dummies(data_df, dtype=int, columns=['Weather_Conditions',
                                                                  'Road_Surface_Conditions',
                                                                  'Vehicle_Manoeuvre'])

    data_df_encoded['column_normalized'] = -1 + (
            (data_df_encoded['Speed_limit'] - data_df_encoded['Speed_limit'].min()) * 2) / (
                                            data_df_encoded['Speed_limit'].max() - data_df_encoded['Speed_limit'].min())

    data_df_encoded.drop(['Speed_limit'], axis=1, inplace=True)

    # save the encoded data
    data_df_encoded.to_csv('data/encoded_data.csv', index=False)
