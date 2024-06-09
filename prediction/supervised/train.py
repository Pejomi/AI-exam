import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from joblib import dump, load


def train(ml_ready):
    data = ml_ready

    X = data.drop('Accident_Severity', axis=1)
    y = data['Accident_Severity']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Save the model
    dump(model, 'prediction/supervised/models/severity_predictor_dt.joblib')

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Decision Tree Accuracy: {accuracy:.2f}')

    probabilities = model.predict_proba(X_test)

    auc_score_ovr = roc_auc_score(y_test, probabilities, multi_class='ovr')
    print("AUC Score (One-vs-Rest):", auc_score_ovr)

    auc_score_ovo = roc_auc_score(y_test, probabilities, multi_class='ovo')
    print("AUC Score (One-vs-One):", auc_score_ovo)

    # Get feature importance
    feature_importances = pd.DataFrame(model.feature_importances_,
                                       index=X_train.columns,
                                       columns=['importance']).sort_values('importance', ascending=False)

    # Extract category base names from feature names
    feature_importances['category'] = feature_importances.index.to_series().apply(
        lambda x: x.split('_')[0] if '_' in x else x)

    # Group by the new category column and sum the importance
    category_importance = feature_importances.groupby('category')['importance'].sum().sort_values(ascending=False)

    # print(feature_importances)
    print(category_importance)
