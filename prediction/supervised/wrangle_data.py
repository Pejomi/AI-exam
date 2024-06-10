import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPRegressor
from joblib import dump, load
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import time

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def wrangle_data(data):
    print("Loading and wrangling data...")

    # index_col = data.iloc[:, 0]

    data_cut = pd.DataFrame(data[
        ['Accident_Severity', 'Time', 'Light_Conditions', 'Weather_Conditions', 'Speed_limit', 'Road_Type',
         'Road_Surface_Conditions', 'Urban_or_Rural_Area', 'Age_Band_of_Driver', 'Sex_of_Driver', 'Age_of_Vehicle',
         'Vehicle_Manoeuvre', 'make', 'Vehicle_Type']])

    # data_cut.loc[:, 'ID'] = index_col

    data_cut.replace('No data', np.nan, inplace=True)
    data_cut.replace('Not known', np.nan, inplace=True)
    data_cut.replace('', np.nan, inplace=True)

    # Drop rows with NaN in all but 2 columns. These are the columns where we want to fill the missing values.
    cols_to_drop_na = [col for col in data_cut.columns if col not in ['Age_of_Vehicle', 'Age_Band_of_Driver']]
    data_cut.dropna(subset=cols_to_drop_na, inplace=True)

    # Convert Time to minutes since midnight
    data_cut['Time'] = pd.to_timedelta(data_cut['Time'] + ':00')
    data_cut['Time'] = data_cut['Time'].dt.total_seconds() / 60

    # Fill NaN values in 'Age_of_Vehicle' with the median value
    data_cut.loc[:, 'Age_of_Vehicle'] = data_cut['Age_of_Vehicle'].fillna(data_cut['Age_of_Vehicle'].median())

    # Fill NaN values in 'Age_Band_of_Driver' with the mode value
    data_cut.loc[:, 'Age_Band_of_Driver'] = data_cut['Age_Band_of_Driver'].fillna(data_cut.mode().iloc[0]['Age_Band_of_Driver'])

    data_cut.to_csv('data/clean_filled.csv', index=False, encoding='utf-8')

    # id_age_df = fill_aov_nan(data_cut.copy())

    # Create a map from id_age_df
    # id_map = id_age_df.set_index('ID')['Age_of_Vehicle']

    # Update data_cut using map
    # data_cut['Age_of_Vehicle'] = data_cut['ID'].map(id_map).fillna(data_cut['Age_of_Vehicle'])

    # data_filled_aov = pd.read_csv('data/with_filled_aov.csv', encoding='utf-8')
    # data_filled_abod = fill_abod_nan(data_filled_aov.copy())
    # data_filled_abod.to_csv('data/with_filled_aov_abod.csv', index=False, encoding='utf-8')



def fill_abod_nan(data):
    # Drop rows with NaN in all but 1 column. This is the columns where we want to predict the missing values.
    cols_to_drop_na = [col for col in data.columns if col not in ['Age_Band_of_Driver']]
    data.dropna(subset=cols_to_drop_na, inplace=True)

    # Collect less frequent makes into 'Other' category
    make_counts = data['make'].value_counts()
    threshold = 10
    other_makes = make_counts[make_counts < threshold].index

    data['make'] = data['make'].replace(other_makes, 'Other')

    print("Data loading and wrangling complete")
    print(data.shape)

    # train_and_save_abod_model(data)

    scaler = load('prediction/supervised/models/scaler_for_abod_predictor.joblib')
    endcoder = load('prediction/supervised/models/encoder_for_abod_predictor.joblib')
    model = load('prediction/supervised/models/abod_predictor.joblib')

    abod_pred_input = data[data['Age_Band_of_Driver'].isna()]
    abod_pred_input = abod_pred_input[['make', 'Vehicle_Type', 'Age_of_Vehicle', 'Sex_of_Driver']]
    abod_pred_input = pd.get_dummies(abod_pred_input, columns=['make', 'Vehicle_Type', 'Sex_of_Driver'])

    # Scale (normalize) the target column
    abod_pred_input.loc[:, 'Age_of_Vehicle'] = scaler.transform(abod_pred_input[['Age_of_Vehicle']])

    # Generate predictions for missing Age Band of Driver values
    predictions_encoded = model.predict(abod_pred_input)

    # Reshape predictions from 1D array to 2D array
    predictions_encoded = predictions_encoded.reshape(-1, 1)

    # Convert predictions from scaled to original values
    predictions = endcoder.inverse_transform(predictions_encoded)

    # Fill NaN values with predictions converted from normalized to original values
    data.loc[data['Age_Band_of_Driver'].isna(), 'Age_Band_of_Driver'] = predictions

    return data


def fill_aov_nan(data):
    # Drop rows with NaN in all but 1 column. This is the columns where we want to predict the missing values.
    cols_to_drop_na = [col for col in data.columns if col not in ['Age_of_Vehicle']]
    data.dropna(subset=cols_to_drop_na, inplace=True)

    # Collect less frequent makes into 'Other' category
    make_counts = data['make'].value_counts()
    threshold = 10
    other_makes = make_counts[make_counts < threshold].index

    data['make'] = data['make'].replace(other_makes, 'Other')

    print("Data loading and wrangling complete")
    print(data.shape)

    # train_and_save_aov_model(data)

    scaler = load('prediction/supervised/models/scaler_for_aov_predictor.joblib')
    model = load('prediction/supervised/models/aov_predictor.joblib')

    aov_pred_input = data[data['Age_of_Vehicle'].isna()]
    aov_pred_input = aov_pred_input[['make', 'Vehicle_Type', 'Age_Band_of_Driver', 'Sex_of_Driver']]
    aov_pred_input = pd.get_dummies(aov_pred_input, columns=['make', 'Vehicle_Type', 'Age_Band_of_Driver', 'Sex_of_Driver'])

    # Generate predictions for missing Age of Vehicle values
    predictions_scaled = model.predict(aov_pred_input)

    # Reshape predictions from 1D array to 2D array
    predictions_scaled = predictions_scaled.reshape(-1, 1)

    # Convert predictions from scaled to original values
    predictions = scaler.inverse_transform(predictions_scaled)

    # Round predictions to nearest whole number
    predictions = [[float(round(num)) for num in row] for row in predictions]

    # Flatten the list of lists into a 1D list
    flattened_pred = [item for sublist in predictions for item in sublist]

    # Create a new DataFrame
    id_age_df = pd.DataFrame({
        'ID': data.loc[data['Age_of_Vehicle'].isna(), 'ID'],
        'Age_of_Vehicle': flattened_pred
    })

    return id_age_df


def train_and_save_aov_model(data):
    print("Training and saving Age of Vehicle predictor model...")

    data = pd.DataFrame(data[['make', 'Vehicle_Type', 'Age_Band_of_Driver', 'Sex_of_Driver', 'Age_of_Vehicle']])

    # One-hot encode the categorical columns
    data = pd.get_dummies(data, columns=['make', 'Vehicle_Type', 'Age_Band_of_Driver', 'Sex_of_Driver'])
    print(data.shape)

    # Drop rows with NaN in the target column
    data = data.dropna(subset=['Age_of_Vehicle'])

    # Scale (normalize) the target column
    scaler = MinMaxScaler()
    data.loc[:, 'Age_of_Vehicle'] = scaler.fit_transform(data[['Age_of_Vehicle']])

    # Save the scaler for later use
    dump(scaler, 'prediction/supervised/models/scaler_for_aov_predictor.joblib')

    X = data.drop('Age_of_Vehicle', axis=1)
    y = data['Age_of_Vehicle']

    # compare_regression_models(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = MLPRegressor(
        hidden_layer_sizes=(25,),
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



def train_and_save_abod_model(data):
    print("Training and saving Age Band of Driver predictor model...")

    cols_to_drop_na = [col for col in data.columns if col not in ['Age_Band_of_Driver']]
    data.dropna(subset=cols_to_drop_na, inplace=True)

    # Collect less frequent makes into 'Other' category
    make_counts = data['make'].value_counts()
    threshold = 10
    other_makes = make_counts[make_counts < threshold].index

    data['make'] = data['make'].replace(other_makes, 'Other')

    data = pd.DataFrame(data[['make', 'Vehicle_Type', 'Sex_of_Driver', 'Age_of_Vehicle', 'Age_Band_of_Driver']])

    # One-hot encode the categorical columns
    data = pd.get_dummies(data, columns=['make', 'Vehicle_Type', 'Sex_of_Driver'])
    print(data.shape)

    # Drop rows with NaN in the target column
    data = data.dropna(subset=['Age_Band_of_Driver'])

    # Scale (normalize) the target column
    scaler = StandardScaler()
    data.loc[:, 'Age_of_Vehicle'] = scaler.fit_transform(data[['Age_of_Vehicle']])

    # Save the scaler for later use
    dump(scaler, 'prediction/supervised/models/scaler_for_abod_predictor.joblib')

    X = data.drop('Age_Band_of_Driver', axis=1)
    y = data['Age_Band_of_Driver']

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    # Save the encoder for later use
    dump(encoder, 'prediction/supervised/models/encoder_for_abod_predictor.joblib')

    # compare_classification_models(X, y_encoded)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    model = Sequential([
        Dense(X_train.shape[1], activation='relu'),
        Dropout(0.5),
        Dense(25, activation='relu'),
        Dropout(0.5),
        Dense(len(np.unique(y_encoded)), activation='softmax')  # Replace with number of classes
    ])

    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=15, batch_size=32, validation_split=0.2, verbose=1)

    dump(model, 'prediction/supervised/models/abod_predictor.joblib')

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Neural Network accuracy: {test_acc:.4f}")



def compare_classification_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "Gradient Boosting": GradientBoostingClassifier(),
        "SVM": make_pipeline(StandardScaler(), SVC(gamma='auto'))
    }

    for name, model in models.items():
        start_time = time.time()
        print(f"Training {name}...")
        model.fit(X_train, y_train)

        print(f"Testing {name}...")
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        print(f"{name}:")
        print(f"Accuracy: {accuracy}")
        print(f"F1: {f1}")
        print(f"Time taken: {time.time() - start_time}")
        print()



def compare_regression_models(X, y):
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
