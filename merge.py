import pandas as pd
import os

# Define paths to the dataset files
assets_folder = os.path.join(os.getcwd(), "assets")
train_path = os.path.join(assets_folder, "train.txt")

# Read the training triple file
train_df = pd.read_csv(train_path, sep='\t', names=['subject', 'predicate', 'object'])

# Ensure the results directory exists
results_folder = os.path.join(os.getcwd(), "results")
os.makedirs(results_folder, exist_ok=True)

# Save the training triples to a CSV file for processing
output_path = os.path.join(results_folder, "all_triples.csv")
train_df.to_csv(output_path, index=False)
print(f"Training dataset saved to: {output_path}")
