import pandas as pd

def save_anomalous_triples(original_dataset_path, svm_output_path, output_path):
    """
    Save all anomalous triples to a file.

    Args:
        original_dataset_path (str): Path to the original dataset file (e.g., 'train.txt').
        svm_output_path (str): Path to the SVM output file (e.g., 'umls_svm_output.pkl').
        output_path (str): Path to save the anomalous triples (e.g., 'anomalous_triples.txt').
    """
    try:
        # Load the original dataset (train.txt)
        print("Loading the original dataset...")
        original_df = pd.read_csv(original_dataset_path, sep='\t', names=['subject', 'predicate', 'object'])

        # Load the SVM output dataset
        print("Loading the SVM output dataset...")
        svm_df = pd.read_pickle(svm_output_path)

        # Ensure 'svm_binary_output' column exists in SVM output
        if 'svm_binary_output' not in svm_df.columns:
            raise ValueError("SVM output must contain 'svm_binary_output' column.")

        # Filter anomalous triples from SVM output
        print("Filtering anomalous triples...")
        anomalous_df = svm_df[svm_df['svm_binary_output'] == -1]

        # Ensure required columns exist in the anomalous dataset
        if not {'subject', 'predicate', 'object'}.issubset(anomalous_df.columns):
            raise ValueError("SVM output must contain 'subject', 'predicate', and 'object' columns.")

        # Extract anomalous triples (subject, predicate, object)
        anomalous_triples = anomalous_df[['subject', 'predicate', 'object']]

        # Save anomalous triples to the output file
        if anomalous_triples.empty:
            print("No anomalous triples found!")
        else:
            print(f"Saving anomalous triples to {output_path}...")
            anomalous_triples.to_csv(output_path, sep='\t', index=False, header=False)
            print(f"Anomalous triples saved to: {output_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    # Paths to the datasets
    original_dataset_path = input("Enter the absolute file path of the original dataset (e.g., 'train.txt'): ").strip()
    svm_output_path = input("Enter the absolute file path of the SVM output (e.g., 'umls_svm_output.pkl'): ").strip()
    output_path = "anomalous_triples.txt"

    # Save the anomalous triples
    save_anomalous_triples(original_dataset_path, svm_output_path, output_path)
