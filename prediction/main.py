import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense, Dropout
from keras.optimizers import Adam


# Load data
data = pd.read_csv('../data/merged_information.csv', nrows=10000)

# Select relevant features
features = [
    '1st_Road_Class', 'Carriageway_Hazards', 'Day_of_Week',
    'Did_Police_Officer_Attend_Scene_of_Accident', 'Junction_Control', 'Junction_Detail',
    'Latitude', 'Longitude', 'Light_Conditions', 'Local_Authority_(District)', 
    'Road_Surface_Conditions', 'Road_Type', 'Speed_limit', 'Time', 'Urban_or_Rural_Area',
    'Weather_Conditions', 'Number_of_Casualties', 'Number_of_Vehicles'
]


target = 'Accident_Severity'

X = data[features]
y = data[target]

# Convert target to numerical categories
severity_mapping = {'Slight': 0, 'Serious': 1, 'Fatal': 2}
y = y.map(severity_mapping)

# Data preprocessing
categorical_features = X.select_dtypes(include=['object', 'category']).columns
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns


preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)])

X_processed = preprocessor.fit_transform(X)


# Convert the sparse matrix to a dense array
X_processed_dense = X_processed.toarray()

# Convert target to categorical
y_categorical = tf.keras.utils.to_categorical(y)

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X_processed_dense, y_categorical, test_size=0.2, random_state=42)


# Build the ANN model
model = Sequential()
model.add(Dense(64, input_dim=X_processed_dense.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(y_categorical.shape[1], activation='softmax'))

model.summary()

# Number of epochs
epochs = 10

# Learning rate
learning_rate = 0.001

# Compile the model
model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])



# Train the model
model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy}")

# Predictions
y_pred = model.predict(X_test)