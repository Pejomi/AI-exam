import pandas as pd
from joblib import dump, load
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OrdinalEncoder

pd.set_option('display.max_columns', None)


def preprocess(clean_filled):

    # Keep only the relevant columns
    data = pd.DataFrame(clean_filled[['Accident_Severity', 'Time', 'Light_Conditions', 'Weather_Conditions', 'Speed_limit', 'Road_Type',
         'Road_Surface_Conditions', 'Urban_or_Rural_Area', 'Age_Band_of_Driver', 'Sex_of_Driver', 'Age_of_Vehicle',
         'Vehicle_Manoeuvre', 'Vehicle_Type']])

    # Find only Car accidents
    data = data[data['Vehicle_Type'] == 'Car']
    data = data.drop('Vehicle_Type', axis=1)
    
    numerical_cols = ['Time', 'Age_of_Vehicle']
    categorical_cols = [col for col in data.columns if col not in numerical_cols and col != 'Accident_Severity']

    # Initialize scalers
    minmax_scaler = MinMaxScaler()
    standard_scaler = StandardScaler()

    # Normalize Time column
    data['Time'] = minmax_scaler.fit_transform(data[['Time']])

    # Standardize Age_of_Vehicle column
    data['Age_of_Vehicle'] = standard_scaler.fit_transform(data[['Age_of_Vehicle']])

    # One-hot encode categorical columns
    data = pd.get_dummies(data, columns=categorical_cols)

    # Define the order of severities
    severity_order = ['Slight', 'Serious', 'Fatal']

    # Initialize the encoder with the specified ordinal order of Accident_Severity
    ordinal_encoder = OrdinalEncoder(categories=[severity_order])

    # Fit and transform the data
    data['Accident_Severity'] = ordinal_encoder.fit_transform(data[['Accident_Severity']])

    # Save the encoder and the scalers
    dump(ordinal_encoder, 'prediction/ann/models/severity_encoder.joblib')
    dump(minmax_scaler, 'prediction/ann/models/severity_minmax_scaler.joblib')
    dump(standard_scaler, 'prediction/ann/models/severity_standard_scaler.joblib')

    # save the encoded data
    data.to_csv('/data/ml_ready.csv', index=False)



def preprocess_input_data(raw_data):
    # Keep only the relevant columns
    data = pd.DataFrame(raw_data[['Time', 'Light_Conditions', 'Weather_Conditions', 'Speed_limit',
                             'Road_Type',
                             'Road_Surface_Conditions', 'Urban_or_Rural_Area', 'Age_Band_of_Driver', 'Sex_of_Driver',
                             'Age_of_Vehicle',
                             'Vehicle_Manoeuvre']])

    numerical_cols = ['Time', 'Age_of_Vehicle']
    categorical_cols = [col for col in data.columns if col not in numerical_cols and col != 'Accident_Severity']

    # Initialize scalers
    minmax_scaler = load('prediction/ann/models/severity_minmax_scaler.joblib')
    standard_scaler = load('prediction/ann/models/severity_standard_scaler.joblib')

    # Normalize Time column
    data['Time'] = minmax_scaler.transform(data[['Time']])

    # Standardize Age_of_Vehicle column
    data['Age_of_Vehicle'] = standard_scaler.transform(data[['Age_of_Vehicle']])

    # One-hot encode categorical columns
    data = pd.get_dummies(data, columns=categorical_cols)

    return data



def get_prediction(input_data):
    data = preprocess_input_data(input_data)

    model = load('prediction/ann/models/severity_predictor_dt.joblib')
    prediction = model.predict(data)

    # Inverse transform the ordinal encoding
    ordinal_encoder = load('prediction/ann/models/severity_encoder.joblib')
    prediction = ordinal_encoder.inverse_transform(prediction)

    return prediction
