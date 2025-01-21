import pandas as pd
from collections import Counter
import csv

def get_user_input():
    dataset_path = input("Enter the absolute file path of the SVM output (e.g., '../results/umls_svm_output.pkl'): ").strip()
    print("")
    print("Select an operation to perform:")
    print("  1 - Print the nodes involved in anomalies")
    print("  2 - Print the anomalous triples")
    print("  3 - Fetch the triples matching an anomalous pattern")
    print("  4 - Fetch the triples based on a specific feature")
    print("")
    try:
        operation_number = int(input("Enter the number corresponding to the desired operation: "))
    except ValueError:
        print("Invalid input. Please enter a number between 1 and 4.")
        return
    fetch_data_from_file(dataset_path, operation_number)

def fetch_data_from_file(dataset_path, operation_number):
    # Load the data
    try:
        if dataset_path.endswith('.csv'):
            df = pd.read_csv(dataset_path)
        elif dataset_path.endswith('.pkl'):
            df = pd.read_pickle(dataset_path)
        else:
            print("Unsupported file format. Please provide a .csv or .pkl file.")
            return
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # Drop unnecessary columns if present
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    # Ensure 'svm_binary_output' column exists
    if 'svm_binary_output' not in df.columns:
        print("The 'svm_binary_output' column was not found in the data.")
        return

    # Filter anomalies
    df_abnormal_all = df.loc[df['svm_binary_output'] == -1]

    if operation_number == 1:
        # Option 1: Print nodes with anomalies
        if 'subject' in df_abnormal_all.columns and 'object' in df_abnormal_all.columns:
            subject_nodes = list(df_abnormal_all["subject"])
            object_nodes = list(df_abnormal_all["object"])
            all_nodes = Counter(subject_nodes + object_nodes)
            sorted_all_nodes = dict(sorted(all_nodes.items(), key=lambda item: item[1], reverse=True))
            print("{:<20} {:<10}".format('Node', 'Anomalous Triple Count'))
            for node, count in sorted_all_nodes.items():
                print("{:<20} {:<10}".format(node, count))
        else:
            print("Columns 'subject' and 'object' not found in the data.")

    elif operation_number == 2:
        # Option 2: Print anomalous triples
        print(df_abnormal_all)

    elif operation_number == 3:
        # Option 3: Fetch triples matching an anomalous pattern
        features = input("Enter the anomalous feature names separated by a comma: ").strip()
        features = [f.strip() for f in features.split(",")]
        feature_patterns = input("Enter the corresponding feature patterns (e.g., 'TFT, FTT'): ").strip()
        feature_patterns = [p.strip() for p in feature_patterns.split(",")]
        replacements = {"F": '0', "T": '1'}
        file_name = "anomalous_patterns.csv"
        with open(file_name, 'w', newline='') as features_file:
            writer = csv.writer(features_file)
            writer.writerow(['Pattern'] + features)
            for feature_pattern in feature_patterns:
                filtered_df = df_abnormal_all.copy()
                feature_pattern_text = [replacements.get(char.upper(), char) for char in feature_pattern]
                patterns_and_columns = dict(zip(features, feature_pattern_text))
                for column, value in patterns_and_columns.items():
                    if column in filtered_df.columns:
                        filtered_df = filtered_df[filtered_df[column].astype(str) == value]
                    else:
                        print(f"Feature '{column}' not found in the data.")
                        break
                else:
                    writer.writerow([feature_pattern] + [patterns_and_columns.get(f, '') for f in features])
                    filtered_df.to_csv(features_file, mode='a', index=False)
                    print(f"Anomalous pattern '{feature_pattern}' saved to '{file_name}'.")
                    continue
                print(f"Skipping pattern '{feature_pattern}' due to missing feature.")
    elif operation_number == 4:
        # Option 4: Fetch triples based on a specific feature
        feature_name = input("Enter the feature name: ").strip()
        feature_value = input("Enter the feature value (e.g., '0' or '1'): ").strip()
        if feature_name in df_abnormal_all.columns:
            filtered_df = df_abnormal_all[df_abnormal_all[feature_name].astype(str) == feature_value]
            print(filtered_df)
        else:
            print(f"Feature '{feature_name}' not found in the data.")

    else:
        print("Invalid operation number entered.")

if __name__ == '__main__':
    get_user_input()
