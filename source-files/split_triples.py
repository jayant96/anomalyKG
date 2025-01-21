import pandas as pd
import os
from sklearn.model_selection import train_test_split

def split_small_dataset():
    # Define paths
    project_root = os.path.dirname(os.getcwd())  # Move up one level from 'source-files'
    results_folder = os.path.join(project_root, "results")
    simplified_triples_path = os.path.join(results_folder, "simplified_triples.txt")

    # Check if the simplified triples file exists
    if not os.path.exists(simplified_triples_path):
        print(f"Error: {simplified_triples_path} does not exist.")
        return

    # Load the simplified triples
    print("Loading simplified triples...")
    triples_df = pd.read_csv(simplified_triples_path, sep='\t', names=['subject', 'predicate', 'object'], header=None)

    # Ensure dataset size is suitable for splitting
    if len(triples_df) < 10:
        print("Dataset too small for meaningful splits. Add more data.")
        return

    # Split into train (70%), validate (15%), test (15%)
    train, temp = train_test_split(triples_df, test_size=0.3, random_state=42)
    validate, test = train_test_split(temp, test_size=0.5, random_state=42)

    # Save the splits
    train.to_csv(os.path.join(results_folder, "train.txt"), sep='\t', index=False, header=False)
    validate.to_csv(os.path.join(results_folder, "validate.txt"), sep='\t', index=False, header=False)
    test.to_csv(os.path.join(results_folder, "test.txt"), sep='\t', index=False, header=False)

    print("Splitting completed.")
    print(f"Train: {len(train)} triples, Validate: {len(validate)} triples, Test: {len(test)} triples.")
    print(f"Files saved in: {results_folder}")

if __name__ == "__main__":
    split_small_dataset()
