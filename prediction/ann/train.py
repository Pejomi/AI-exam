import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from joblib import dump, load
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns




def train(ml_ready):
    data = ml_ready

    X = data.drop('Accident_Severity', axis=1)
    y = data['Accident_Severity']

    # Convert target to categorical
    y_categorical = tf.keras.utils.to_categorical(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

    # Building the ANN model
    model = Sequential()
    model.add(Dense(64, input_dim=X.shape[1], activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(y_categorical.shape[1], activation='softmax'))

    # Number of epochs
    epochs = 10

    # Learning rate
    learning_rate = 0.001

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

    # Fit the model
    model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_split=0.2)

    # Save the model
    dump(model, 'ann/models/severity_predictor_dt.joblib')

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy}")

    # Predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)

    # Calculate and print accuracy
    accuracy = accuracy_score(y_test_classes, y_pred_classes)
    print(f"Accuracy: {accuracy:.2f}")

    # Check if all classes are present in y_test before calculating ROC AUC
    if len(np.unique(y_test_classes)) == y_categorical.shape[1]:
        # Calculate ROC AUC scores
        auc_score_ovr = roc_auc_score(y_test, y_pred, multi_class='ovr')
        print("AUC Score (One-vs-Rest):", auc_score_ovr)

        auc_score_ovo = roc_auc_score(y_test, y_pred, multi_class='ovo')
        print("AUC Score (One-vs-One):", auc_score_ovo)
    else:
        print("Not all classes are present in y_test; ROC AUC score is not defined.")

    # Additional evaluations
    print("\nClassification Report:")
    print(classification_report(y_test_classes, y_pred_classes, zero_division=1))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test_classes, y_pred_classes)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', cbar=False)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()