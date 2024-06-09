import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.feature_extraction import FeatureHasher
from joblib import dump, load
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import RandomizedSearchCV
import time

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def wrangle_data(data):
    print("Loading and wrangling data...")

    data_cut = pd.DataFrame(data[
        ['Accident_Severity', 'Time', 'Light_Conditions', 'Weather_Conditions', 'Speed_limit', 'Road_Type',
         'Road_Surface_Conditions', 'Urban_or_Rural_Area', 'Age_Band_of_Driver', 'Sex_of_Driver', 'Age_of_Vehicle',
         'Vehicle_Manoeuvre', 'make', 'Vehicle_Type']])

    categorical_cols = [col for col in data_cut.columns if col not in ['Time', 'Age_of_Vehicle']]
    numerical_cols = ['Time', 'Age_of_Vehicle']

    data_cut.replace('No data', np.nan, inplace=True)
    data_cut.replace('Not known', np.nan, inplace=True)
    data_cut.replace('', np.nan, inplace=True)

    # Drop rows with NaN in all but 1 column. This is the columns where we want to predict the missing values.
    cols_to_drop_na = [col for col in data_cut.columns if col not in ['Age_of_Vehicle']]
    data_cut.dropna(subset=cols_to_drop_na, inplace=True)

    # Convert Time to minutes since midnight
    data_cut['Time'] = pd.to_timedelta(data_cut['Time'] + ':00')
    data_cut['Time'] = data_cut['Time'].dt.total_seconds() / 60

    # Collect less frequent makes into 'Other' category
    make_counts = data_cut['make'].value_counts()
    threshold = 500
    other_makes = make_counts[make_counts < threshold].index

    data_cut['make'] = data_cut['make'].replace(other_makes, 'Other')

    print("Data loading and wrangling complete")
    print(data_cut.shape)

    train_and_save_aov_model(data_cut)

    # scaler = load('prediction/supervised/models/scaler_for_aov_predictor.joblib')
    # model = load('prediction/supervised/models/aov_predictor.joblib')
    #
    # aov_pred_input_df = data_cut[data_cut['Age_of_Vehicle'].isna()]
    # aov_pred_input_df = aov_pred_input_df[['make', 'Vehicle_Type', 'Age_Band_of_Driver', 'Sex_of_Driver']]
    # aov_pred_input_df = pd.get_dummies(aov_pred_input_df, columns=['make', 'Vehicle_Type', 'Age_Band_of_Driver', 'Sex_of_Driver'])
    #
    # print(aov_pred_input_df.isna().sum())
    #
    # predictions = model.predict(aov_pred_input_df)
    #
    # # Fill NaN values with predictions converted from normalized to original values
    # data_cut.loc[data_cut['Age_of_Vehicle'].isna(), 'Age_of_Vehicle'] = scaler.inverse_transform(predictions)
    #
    # print(data_cut.isna().sum())



def train_and_save_aov_model(data):
    print("Training and saving Age of Vehicle predictor model...")

    data_for_aov_pred = pd.DataFrame(data[['make', 'Vehicle_Type', 'Age_Band_of_Driver', 'Sex_of_Driver', 'Age_of_Vehicle']])

    # One-hot encode the categorical columns
    one_hot_aov = pd.get_dummies(data_for_aov_pred, columns=['make', 'Vehicle_Type', 'Age_Band_of_Driver', 'Sex_of_Driver'])

    # Drop rows with NaN in the target column
    one_hot_aov_dropna = one_hot_aov.dropna(subset=['Age_of_Vehicle'])

    # Scale (normalize) the target column
    scaler = MinMaxScaler()
    one_hot_aov_dropna.loc[:, 'Age_of_Vehicle'] = scaler.fit_transform(one_hot_aov_dropna[['Age_of_Vehicle']])

    # Save the scaler for later use
    dump(scaler, 'prediction/supervised/models/scaler_for_aov_predictor.joblib')

    X = one_hot_aov_dropna.drop('Age_of_Vehicle', axis=1)
    y = one_hot_aov_dropna['Age_of_Vehicle']

    # compare_continuous_models(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = MLPRegressor(
        hidden_layer_sizes=(50,),
        max_iter=1000,
        early_stopping=True,
        n_iter_no_change=20
    )
    print("Training Age of Vehicle predictor model...")
    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()

    # Save the model for later use
    print("Saving Age of Vehicle predictor model...")
    dump(model, 'prediction/supervised/models/aov_predictor.joblib')

    y_pred = model.predict(X_test)

    print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
    print(f"R-squared: {r2_score(y_test, y_pred)}")
    print(f"Time taken: {end_time - start_time}")


def compare_continuous_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'Linear Regression': LinearRegression(),
        # Random Forest is disabled due to poor performance and long training time
        'Random Forest': RandomForestRegressor(),
        'Gradient Boosting': GradientBoostingRegressor(),
        # SVR is disabled due to long training time (10,000 rows, 2.5 sec. | 40,000 rows, 43 sec.)
        # 'Support Vector Machine': SVR(),
        'Neural Network': MLPRegressor()
    }

    for name, model in models.items():
        start_time = time.time()
        print(f"Training {name}...")
        model.fit(X_train, y_train)

        print(f"Testing {name}...")
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"{name}:")
        print(f"Mean Squared Error: {mse}")
        print(f"R-squared: {r2}")
        print(f"Time taken: {time.time() - start_time}")
        print()


def calculate_midpoint(range_str):
    if ' - ' in range_str:
        # It's a range, calculate the midpoint
        lower, upper = map(int, range_str.split(' - '))
        return (lower + upper) / 2
    else:
        # It's a single value (e.g., "80" for "Over 75"), directly convert to int
        return int(range_str)
