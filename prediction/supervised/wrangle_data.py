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

    # data_for_aov_pred = pd.DataFrame(data[['make', 'model', 'Sex_of_Driver', 'Age_of_Vehicle']])
    data_for_aov_pred = pd.DataFrame(data[['make', 'Vehicle_Type', 'Age_Band_of_Driver', 'Sex_of_Driver', 'Age_of_Vehicle']])

    data_for_aov_pred.replace('No data', np.nan, inplace=True)
    data_for_aov_pred.replace('Not known', np.nan, inplace=True)
    data_for_aov_pred.replace('', np.nan, inplace=True)

    cols_to_drop_na = [col for col in data_for_aov_pred.columns if col not in ['Age_of_Vehicle']]
    data_for_aov_pred.dropna(subset=cols_to_drop_na, inplace=True)

    data_cut = pd.DataFrame(data[
        ['Accident_Severity', 'Time', 'Light_Conditions', 'Weather_Conditions', 'Speed_limit', 'Road_Type',
         'Road_Surface_Conditions', 'Urban_or_Rural_Area', 'Age_Band_of_Driver', 'Sex_of_Driver', 'Age_of_Vehicle',
         'Vehicle_Manoeuvre']])

    categorical_cols = [col for col in data_cut.columns if col not in ['Time', 'Age_of_Vehicle']]
    numerical_cols = ['Time', 'Age_of_Vehicle']

    data_cut.replace('No data', np.nan, inplace=True)
    data_cut.replace('', np.nan, inplace=True)

    # Drop rows with NaN in all but 2 columns. These 2 are the columns where we want to predict the missing values.
    cols_to_drop_na = [col for col in data_cut.columns if col not in ['Age_Band_of_Driver', 'Age_of_Vehicle']]
    data_cut.dropna(subset=cols_to_drop_na, inplace=True)

    # Convert Time to minutes since midnight
    data_cut['Time'] = pd.to_timedelta(data_cut['Time'] + ':00')
    data_cut['Time'] = data_cut['Time'].dt.total_seconds() / 60

    print("Data loading and wrangling complete")

    # Firsly, the empty values of 'Age_of_Vehicle' will be predicted
    one_hot_aov = pd.get_dummies(data_for_aov_pred, columns=['make', 'Vehicle_Type', 'Age_Band_of_Driver', 'Sex_of_Driver'])
    # dicts = data_for_aov_pred[['make', 'model', 'Sex_of_Driver']].to_dict(orient='records')
    # hasher = FeatureHasher(n_features=1000, input_type='string')
    # hashed_aov = hasher.transform(dicts)
    # hashed_aov_df = pd.DataFrame(hashed_aov.toarray())
    # hashed_aov_df['Age_of_Vehicle'] = data_for_aov_pred['Age_of_Vehicle']
    #
    one_hot_aov_dropna = one_hot_aov.dropna(subset=['Age_of_Vehicle'])
    print(one_hot_aov_dropna.shape)

    scaler = MinMaxScaler()
    one_hot_aov_dropna.loc[:, 'Age_of_Vehicle'] = scaler.fit_transform(one_hot_aov_dropna[['Age_of_Vehicle']])

    X = one_hot_aov_dropna.drop('Age_of_Vehicle', axis=1)
    y = one_hot_aov_dropna['Age_of_Vehicle']

    compare_continuous_models(X, y)

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    #
    # model = RandomForestRegressor()
    # model.fit(X_train, y_train)
    #
    # y_pred = model.predict(X_test)
    #
    # print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
    # print(f"R-squared: {r2_score(y_test, y_pred)}")


    # predictions = model.predict(X_test)

    # Fill NaN values with predictions
    # one_hot_aov.loc[one_hot_aov['Age_of_Vehicle'].isna(), 'Age_of_Vehicle'] = predictions


    # print("Before drop NaN sum:\n", data_cut.isna().sum())
    # data_cut_cleaned = data_cut.dropna(how='any')
    # print("After drop NaN sum:\n", data_cut_cleaned.isna().sum())
    # print("Length after second drop:", len(data_cut_cleaned))

    # data_cut_cleaned.to_csv('data/cleaned_data.csv', index=False)


def compare_continuous_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_dist = {
        'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
        'activation': ['relu', 'tanh', 'logistic'],
        'solver': ['sgd', 'adam'],
        'alpha': 10.0 ** -np.arange(1, 7),
        'batch_size': [32, 64, 128, 'auto'],
        'learning_rate_init': 10.0 ** -np.arange(3, 6),
        'max_iter': [200, 400, 800],
        'early_stopping': [True, False],
        'validation_fraction': [0.1, 0.2],
        'momentum': [0.9, 0.95],
        'n_iter_no_change': [10, 20]
    }

    # mlp = MLPRegressor()
    # random_search = RandomizedSearchCV(mlp, param_distributions=param_dist, n_iter=100, cv=3, verbose=2,
    #                                    random_state=42, n_jobs=-1)
    # random_search.fit(X_train, y_train)
    #
    # print("Best parameters found: ", random_search.best_params_)

    models = {
        # 'Linear Regression': LinearRegression(),
        # Random Forest is disabled due to poor performance and long training time
        # 'Random Forest': RandomForestRegressor(),
        # 'Gradient Boosting': GradientBoostingRegressor(),
        # SVR is disabled due to long training time (10,000 rows, 2.5 sec. | 40,000 rows, 43 sec.)
        # 'Support Vector Machine': SVR(),
        'Neural Network': MLPRegressor(
        hidden_layer_sizes=(100, 50),
        activation='tanh',
        solver='adam',
        alpha=0.001,
        batch_size=64,
        learning_rate_init=0.0001,
        max_iter=400,
        shuffle=True,
        random_state=None,
        tol=1e-4,
        verbose=False,
        warm_start=False,
        momentum=0.95,
        nesterovs_momentum=True,
        early_stopping=False,
        validation_fraction=0.2,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
        n_iter_no_change=10,
        max_fun=15000
    )
    }

    for name, model in models.items():
        start_time = time.time()
        print(f"Training {name}...")
        model.fit(X_train, y_train)

        # dump(model, 'prediction/supervised/models/aov_predictor.joblib')

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
