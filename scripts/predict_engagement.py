import json
import pandas as pd
import numpy as np
import joblib
import os
import sys
import warnings

warnings.filterwarnings("ignore", message="X does not have valid feature names")


def get_script_directory():
    return os.path.dirname(os.path.abspath(__file__))


def load_session(file_path):
    if not os.path.exists(file_path):
        print(f"Error: The session file '{file_path}' does not exist.")
        sys.exit(1)
    with open(file_path, 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data)


def extract_features(df, page_width=800, page_height=600):
    features = {}
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    df['delta_x'] = df['x'].diff().fillna(0)
    df['delta_y'] = df['y'].diff().fillna(0)
    df['delta_time'] = df['timestamp'].diff().fillna(0) / 1000 

    df['distance'] = np.sqrt(df['delta_x']**2 + df['delta_y']**2)
    total_distance = df['distance'].sum()
    features['total_distance'] = total_distance

    total_time = df['timestamp'].iloc[-1] / 1000 
    average_speed = total_distance / total_time if total_time > 0 else 0
    features['average_speed'] = average_speed

    df['speed'] = df['distance'] / df['delta_time'].replace(0, np.nan)
    speed_std = df['speed'].std(skipna=True)
    features['speed_std'] = speed_std if not np.isnan(speed_std) else 0

    movement_threshold = 5  
    num_movements = (df['distance'] > movement_threshold).sum()
    features['num_movements'] = num_movements

    idle_time = df[df['distance'] == 0]['delta_time'].sum()
    features['idle_time'] = idle_time

    df['angle'] = np.degrees(np.arctan2(df['delta_y'], df['delta_x']))
    df['angle_diff'] = df['angle'].diff().fillna(0).abs()
    direction_change_threshold = 30  
    direction_changes = (df['angle_diff'] > direction_change_threshold).sum()
    features['direction_changes'] = direction_changes

    x_min, x_max = df['x'].min(), df['x'].max()
    y_min, y_max = df['y'].min(), df['y'].max()
    x_range = x_max - x_min if x_max != x_min else 1
    y_range = y_max - y_min if y_max != y_min else 1
    page_area = x_range * y_range
    unique_positions = df[['x', 'y']].drop_duplicates().shape[0]
    coverage = unique_positions / page_area
    features['coverage'] = coverage

    return features


def load_model(model_path):
    if not os.path.exists(model_path):
        print(f"Error: The model file '{model_path}' does not exist.")
        sys.exit(1)
    model = joblib.load(model_path)
    return model


def predict_engagement(model, features):
    feature_order = ['total_distance', 'average_speed', 'speed_std', 'num_movements',
                     'idle_time', 'direction_changes', 'coverage']
    try:
        feature_values = [features[feat] for feat in feature_order]
    except KeyError as e:
        print(f"Error: Missing feature '{e.args[0]}' in the extracted features.")
        sys.exit(1)
    
    input_data = np.array([feature_values])
    
    prediction = model.predict(input_data)
    
    return prediction[0]


def main():
    script_dir = get_script_directory()
    session_file = os.path.join(script_dir, '..', 'session.json')
    model_path = os.path.join(script_dir, '..', 'cursor_classifier.pkl')
    
    model = load_model(model_path)
    
    df = load_session(session_file)
    
    features = extract_features(df)
    
    prediction = predict_engagement(model, features)
    
    status = 'Engaged' if prediction == 1 else 'Disengaged'
    print(f"Prediction: {status}")


if __name__ == "__main__":
    main()
