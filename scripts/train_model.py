import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
import sys

def load_data(csv_path):
    if not os.path.exists(csv_path):
        print(f"Error: The file '{csv_path}' does not exist.")
        sys.exit(1)
    df = pd.read_csv(csv_path)
    return df

def prepare_data(df):
    X = df.drop('label', axis=1)  
    y = df['label']              
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, n_estimators=100, random_state=42):
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    clf.fit(X_train, y_train)
    return clf

def evaluate_model(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Disengaged', 'Engaged'])
    print(f"Accuracy: {acc * 100:.2f}%")
    print("Classification Report:")
    print(report)

def save_model(clf, model_path):
    joblib.dump(clf, model_path)
    print(f"Model saved to '{model_path}'")

def main():
    base_dir = os.path.abspath(os.path.dirname(__file__))
    csv_path = os.path.join(base_dir, '..', 'cursor_features.csv')
    model_path = os.path.join(base_dir, '..', 'cursor_classifier.pkl')

    print(f"Loading data from: {csv_path}")
    df = load_data(csv_path)

    X, y = prepare_data(df)

    X_train, X_test, y_train, y_test = split_data(X, y)

    print("Training model...")
    clf = train_model(X_train, y_train)

    print("Evaluating model...")
    evaluate_model(clf, X_test, y_test)

    save_model(clf, model_path)

if __name__ == "__main__":
    main()
