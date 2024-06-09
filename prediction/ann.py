import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# import keras
# from keras.models import Sequential
# from keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense, Dropout
# from keras.optimizers import Adam

# Load data
data = pd.read_csv('../data/merged_information_clean.csv')

# Only Car accidents
data = data[data['Vehicle_Type'] == 'Car']

df = data.dropna()


# Select relevant features
features = [
    #"Time",
    "Light_Conditions",
    "Weather_Conditions",
    "Speed_limit",
    "Road_Type",
    "Road_Surface_Conditions",
    "Urban_or_Rural_Area",
    "Age_Band_of_Driver",
    "Sex_of_Driver",
    "Age_of_Vehicle",
    "Vehicle_Manoeuvre"
]


target = 'Accident_Severity'

X = df[features]
y = df[target]

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

# Building the ANN model
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

model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy}")

# Predictions
y_pred = model.predict(X_test)

if len(y_pred[0]) > 2:  # Multi-class classification
    from sklearn.metrics import roc_auc_score

    # Calculate AUC score
    auc_score = roc_auc_score(y_test, y_pred, multi_class='ovr')
    print(f"AUC Score: {auc_score}")
