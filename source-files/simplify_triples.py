import pandas as pd
import os

def simplify_triples():
    # Define paths
    project_root = os.path.dirname(os.getcwd())  # Move up one level from 'source-files'
    results_folder = os.path.join(project_root, "results")
    cleaned_triples_path = os.path.join(results_folder, "cleaned_all_triples.csv")
    simplified_triples_path = os.path.join(results_folder, "simplified_triples.txt")

    # Check if the cleaned triples file exists
    if not os.path.exists(cleaned_triples_path):
        print(f"Error: {cleaned_triples_path} does not exist.")
        return

    # Load the cleaned triples file
    print("Loading cleaned triples...")
    cleaned_triples_df = pd.read_csv(cleaned_triples_path)

    # Keep only the subject, predicate, and object columns
    if not {'subject', 'predicate', 'object'}.issubset(cleaned_triples_df.columns):
        print("Error: The required columns ('subject', 'predicate', 'object') are missing from the file.")
        return

    simplified_triples_df = cleaned_triples_df[['subject', 'predicate', 'object']]

    # Save to a tab-separated file
    print(f"Saving simplified triples to {simplified_triples_path}...")
    simplified_triples_df.to_csv(simplified_triples_path, sep='\t', index=False, header=False)
    print("Simplification completed.")

if __name__ == "__main__":
    simplify_triples()
