import pandas as pd
import os

def remove_anomalous_triples():
    # Define paths relative to your project structure
    root_folder = os.path.abspath(os.path.join(os.getcwd(), ".."))
    results_folder = os.path.join(root_folder, "results")
    
    all_triples_path = os.path.join(results_folder, "all_triples.csv")
    anomalies_path = os.path.join(results_folder, "umls_svm_output.pkl")
    output_cleaned_path = os.path.join(results_folder, "cleaned_all_triples.csv")

    # Check if files exist
    if not os.path.exists(all_triples_path):
        print(f"Error: {all_triples_path} does not exist.")
        return
    if not os.path.exists(anomalies_path):
        print(f"Error: {anomalies_path} does not exist.")
        return

    # Load data
    print("Loading data...")
    all_triples_df = pd.read_csv(all_triples_path)
    anomalies_df = pd.read_pickle(anomalies_path)

    # Identify the row indices that are anomalies (svm_binary_output == -1)
    # NOTE: This assumes anomalies_df.index matches the original row order of all_triples_df
    print("Filtering anomalies by index...")
    anomalies_idx = anomalies_df.index[anomalies_df['svm_binary_output'] == -1]

    # Drop those indices from your all_triples DataFrame
    cleaned_triples_df = all_triples_df.drop(index=anomalies_idx, errors='ignore')

    # Save the cleaned DataFrame
    print(f"Saving cleaned triples to {output_cleaned_path}...")
    cleaned_triples_df.to_csv(output_cleaned_path, index=False)
    print("Operation completed. Anomalous triples removed.")

if __name__ == "__main__":
    remove_anomalous_triples()
