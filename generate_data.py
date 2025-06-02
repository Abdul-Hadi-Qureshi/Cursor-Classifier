import pandas as pd
import numpy as np

# Generate synthetic data based on trends in the dataset
def generate_synthetic_data(existing_data, num_samples):
    np.random.seed(42)  # For reproducibility

    synthetic_data = []
    for _ in range(num_samples):
        sample = {
            "total_distance": np.random.uniform(500, 27000),
            "average_speed": np.random.uniform(10, 620),
            "speed_std": np.random.uniform(150, 600),
            "num_movements": np.random.randint(5, 550),
            "idle_time": np.random.uniform(0.5, 60),
            "direction_changes": np.random.randint(4, 530),
            "coverage": np.random.uniform(0.00005, 0.002),
            "label": np.random.choice([0, 1], p=[0.7, 0.3])  # Assume 70% label 0, 30% label 1
        }
        synthetic_data.append(sample)

    return pd.DataFrame(synthetic_data)

# Load existing data
data = pd.read_csv("cursor_features.csv")

# Generate 100 new synthetic samples
new_data = generate_synthetic_data(data, 100)

# Combine existing and synthetic data
combined_data = pd.concat([data, new_data], ignore_index=True)

# Save combined data to CSV
combined_data.to_csv("cursor_features.csv", index=False)

print("Synthetic data generation complete. Augmented dataset saved as 'augmented_data.csv'.")
