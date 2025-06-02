import os
import json
import pandas as pd
import numpy as np

ENGAGED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'engaged'))
DISENGAGED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'disengaged'))

def load_session(file_path):
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
    features['average_speed'] = total_distance / total_time if total_time > 0 else 0

    df['speed'] = df['distance'] / df['delta_time'].replace(0, np.nan)
    speed_std = df['speed'].std(skipna=True)
    features['speed_std'] = speed_std if not np.isnan(speed_std) else 0

    movement_threshold = 5
    features['num_movements'] = (df['distance'] > movement_threshold).sum()

    features['idle_time'] = df[df['distance'] == 0]['delta_time'].sum()

    df['angle'] = np.degrees(np.arctan2(df['delta_y'], df['delta_x']))
    df['angle_diff'] = df['angle'].diff().fillna(0).abs()
    direction_change_threshold = 30
    features['direction_changes'] = (df['angle_diff'] > direction_change_threshold).sum()

    x_min, x_max = df['x'].min(), df['x'].max()
    y_min, y_max = df['y'].min(), df['y'].max()
    x_range = x_max - x_min if x_max != x_min else 1
    y_range = y_max - y_min if y_max != y_min else 1
    unique_positions = df[['x', 'y']].drop_duplicates().shape[0]
    features['coverage'] = unique_positions / (x_range * y_range)

    return features

def process_data(directory, label):
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")

    feature_list = []
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)
            df = load_session(file_path)
            features = extract_features(df)
            features['label'] = label
            feature_list.append(features)
    return feature_list

def main():
    try:
        print(f"Processing engaged data from: {ENGAGED_DIR}")
        engaged_features = process_data(ENGAGED_DIR, 1)
        
        print(f"Processing disengaged data from: {DISENGAGED_DIR}")
        disengaged_features = process_data(DISENGAGED_DIR, 0)
        
        all_features = engaged_features + disengaged_features
        df_features = pd.DataFrame(all_features)
        df_features.fillna(0, inplace=True)

        output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'cursor_features.csv'))
        df_features.to_csv(output_path, index=False)
        print(f'Feature extraction completed. Data saved to {output_path}')
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
