import os

def load_triples(file_path):
    """
    Load triples from a file and return as a set.
    Each line is expected to be a triple with tab-separated values.
    """
    with open(file_path, 'r') as file:
        return set(line.strip() for line in file if line.strip())

def find_noisy_triples(clean_file, noisy_file):
    """
    Identify triples in the noisy dataset that are not in the clean dataset.
    """
    clean_triples = load_triples(clean_file)
    noisy_triples = load_triples(noisy_file)
    return sorted(list(noisy_triples - clean_triples))

def main():
    # Prompt user for file paths
    clean_file_path = input("Enter the path to the original (clean) dataset: ").strip()
    noisy_file_path = input("Enter the path to the noisy dataset: ").strip()
    
    # Validate the provided paths
    if not os.path.exists(clean_file_path):
        print(f"Error: The file '{clean_file_path}' does not exist.")
        return
    if not os.path.exists(noisy_file_path):
        print(f"Error: The file '{noisy_file_path}' does not exist.")
        return
    
    # Find noisy triples
    noisy_triples = find_noisy_triples(clean_file_path, noisy_file_path)
    
    # Define output path within the `results` folder
    results_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
    os.makedirs(results_folder, exist_ok=True)
    output_file_path = os.path.join(results_folder, "originalnoisytriples.txt")
    
    # Save noisy triples to the results folder
    with open(output_file_path, 'w') as output_file:
        for triple in noisy_triples:
            output_file.write(triple + "\n")
    
    print(f"Found {len(noisy_triples)} noisy triples. Saved to '{output_file_path}'.")

if __name__ == "__main__":
    main()
