import csv
import pandas as pd
from collections import Counter
import spacy
import networkx as nx
import numpy as np
from nltk.tag import StanfordNERTagger

# File paths
csv_links = "../results/all_triples.csv"
entity_file = "../results/entities_links.csv"
csv_with_features = "../results/features_selected_links.csv"

def construct_df():
    print("Reading triples from all_triples.csv")
    try:
        df_links = pd.read_csv(csv_links)
        print(f"Loaded {len(df_links)} triples from {csv_links}")
    except FileNotFoundError:
        raise Exception(f"File {csv_links} not found. Ensure it exists and is correctly placed.")
    
    # Proceed with entity recognition and other features
    entity_recognition()
    entity_predicate_occur()
    subject_out_degree()

def subject_out_degree():
    print("Calculating Subject Out-Degree")
    df_links = pd.read_csv(csv_links)
    df_links['SubjectOutDeg'] = df_links['subject'].map(Counter(list(df_links['subject'])))
    print("Updated Subject Out-Degree")
    df_links.to_csv(csv_links, index=False)
    subject_in_degree()

def subject_in_degree():
    print("Calculating Subject In-Degree")
    df_links = pd.read_csv(csv_links)
    df_links['SubjectInDeg'] = df_links['subject'].map(Counter(list(df_links['object'])))
    print("Updated Subject In-Degree")
    df_links.to_csv(csv_links, index=False)
    object_out_degree()

def object_out_degree():
    print("Calculating Object Out-Degree")
    df_links = pd.read_csv(csv_links)
    df_links['ObjectOutDeg'] = df_links['object'].map(Counter(list(df_links['subject'])))
    print("Updated Object Out-Degree")
    df_links.to_csv(csv_links, index=False)
    object_in_degree()

def object_in_degree():
    print("Calculating Object In-Degree")
    df_links = pd.read_csv(csv_links)
    df_links['ObjectInDeg'] = df_links['object'].map(Counter(list(df_links['object'])))
    print("Updated Object In-Degree")
    df_links.to_csv(csv_links, index=False)

def entity_recognition():
    print("Running Entity Recognition for FB15K")

    # Load a general-purpose English spaCy model
    import spacy
    nlp = spacy.load("en_core_web_sm")

    # Read the CSV
    df_links = pd.read_csv(csv_links)
    
    # A dictionary to store entity types
    dict_entity_type = {}
    
    # Get all unique entities from subject and object columns
    entities = set(df_links['subject']).union(set(df_links['object']))
    total_entities = len(entities)
    print(f"Total entities to process: {total_entities}")

    # Open output file
    with open(entity_file, "w", newline='', encoding="utf-8") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["entity", "entity_type"])
        count = 0

        for entity in entities:
            count += 1
            print(f"Processing entity {count}/{total_entities}")

            # If FB15K entity is something like /m/02mmwk, extract the last part
            # e.g. "02mmwk", then maybe replace underscores or do additional cleaning
            # For demonstration, do a basic replace:
            entity_split = entity.split("/")[-1]  # e.g. "02mmwk"
            
            # You can further process entity_split if you have a label dictionary.
            # For now, let's attempt to treat it as text:
            entity_final = entity_split.replace("_", " ").strip()

            # Run spaCy NER on this text
            doc = nlp(entity_final)
            
            # Collect spaCy entity labels
            entity_types = [ent.label_ for ent in doc.ents]

            # If spaCy found nothing, fallback:
            if not entity_types:
                entity_types = ["FREEBASE_ENTITY"]  # or "UNCLASSIFIED_ENTITY"

            # Save to dictionary
            dict_entity_type[entity] = set(entity_types)
            
            # Write to CSV
            writer.writerow([entity, set(entity_types)])

    # Assign the recognized types to the DataFrame
    df_links['SubjectEntityType'] = df_links['subject'].map(dict_entity_type)
    df_links['ObjectEntityType'] = df_links['object'].map(dict_entity_type)

    # Save the updated CSV
    df_links.to_csv(csv_links, index=False)
    print("Entity Recognition Complete")


def entity_predicate_occur():
    print("Calculating Entity-Predicate Occurrence")
    df_links = pd.read_csv(csv_links)
    total_groups = df_links.groupby(['SubjectEntityType', 'predicate', 'ObjectEntityType'])
    label_count = {}
    for group in total_groups.groups:
        label_count[group] = len(total_groups.get_group(group))
    df_links['EntityPredOccur'] = df_links.set_index(['SubjectEntityType', 'predicate', 'ObjectEntityType']).index.map(label_count.get)
    df_links.to_csv(csv_links, index=False)
    pred_occur()

def pred_occur():
    print("Calculating Predicate Occurrence")
    df_links = pd.read_csv(csv_links)
    df_links['PredOccur'] = df_links['predicate'].map(Counter(list(df_links['predicate'])))
    df_links.to_csv(csv_links, index=False)
    subj_pred_occur()

def subj_pred_occur():
    print("Calculating Subject-Predicate Occurrence")
    df_links = pd.read_csv(csv_links)
    total_groups = df_links.groupby(['subject', 'predicate'])
    label_count = {}
    for group in total_groups.groups:
        label_count[group] = len(total_groups.get_group(group))
    df_links['SubPredOccur'] = df_links.set_index(['subject', 'predicate']).index.map(label_count.get)
    df_links.to_csv(csv_links, index=False)
    subj_obj_occur()

def subj_obj_occur():
    print("Calculating Subject-Object Occurrence")
    df_links = pd.read_csv(csv_links)
    total_groups = df_links.groupby(['subject', 'object'])
    label_count = {}
    for group in total_groups.groups:
        label_count[group] = len(total_groups.get_group(group))
    df_links['SubObjOccur'] = df_links.set_index(['subject', 'object']).index.map(label_count.get)
    df_links.to_csv(csv_links, index=False)
    dup_triples()

def dup_triples():
    print("Calculating Duplicate Triples")
    df_links = pd.read_csv(csv_links)
    total_groups = df_links.groupby(['subject', 'predicate', 'object'])
    label_count = {}
    for group in total_groups.groups:
        label_count[group] = len(total_groups.get_group(group))
    df_links['DupTriples'] = df_links.set_index(['subject', 'predicate', 'object']).index.map(label_count.get)
    df_links.to_csv(csv_links, index=False)
    corroborative_paths()

def corroborative_paths():
    #this method checks for the existence of alternative knowledge paths.
    print("inside corroborative_paths")
    df_links = pd.read_csv(csv_links)
    label_count = {}
    G = nx.DiGraph()
    for index, rows in df_links.iterrows():
        G.add_edges_from([(rows["subject"],rows["object"])])
        print(rows["subject"],rows["object"])
    for index, rows in df_links.iterrows():
        print("inside corroborative paths")
        count = len(list(nx.all_simple_paths(G, rows["subject"], rows["object"], cutoff=3)))
        print(rows["subject"],rows["object"], count)
        label_count[(rows["subject"], rows["object"])] = count
    df_links['CorrPaths'] = df_links.set_index(['subject', 'object']).index.map(label_count.get)
    df_links.to_csv(csv_links, index=False)
    pred_entity_type_occur()

def pred_entity_type_occur():
#this method counts the no. of times a particular predicate occurs with the two given entity types of subject and object
    print("inside pred_entity_type_occur")
    df_links = pd.read_csv(csv_links)
    total_groups = df_links.groupby(['SubjectEntityType','ObjectEntityType','predicate'])
    label_count = {}
    for group in total_groups.groups:
        label_count[group] = len(total_groups.get_group(group))
    df_links['PredEntityOccur'] = df_links.set_index(['SubjectEntityType','ObjectEntityType','predicate']).index.map(label_count.get)
    df_links.to_csv(csv_links, index=False)
    pred_entity_type_occur_with_source()

def pred_entity_type_occur_with_source():
#this method counts the no. of times a particular predicate occurs with the subject entity types
    print("inside pred_entity_type_occur_with_source")
    df_links = pd.read_csv(csv_links)
    total_groups = df_links.groupby(['SubjectEntityType','predicate'])
    label_count = {}
    for group in total_groups.groups:
        label_count[group] = len(total_groups.get_group(group))
    df_links['PredEntityTypeWithSubject'] = df_links.set_index(['SubjectEntityType','predicate']).index.map(label_count.get)
    df_links.to_csv(csv_links, index=False)
    pred_entity_type_occur_with_target()

def pred_entity_type_occur_with_target():
#this method counts the no. of times a particular predicate occurs with the object entity types
    print("inside pred_entity_type_occur_with_target")
    df_links = pd.read_csv(csv_links)
    total_groups = df_links.groupby(['ObjectEntityType','predicate'])
    label_count = {}
    for group in total_groups.groups:
        label_count[group] = len(total_groups.get_group(group))
    df_links['PredEntityTypeWithObject'] = df_links.set_index(['ObjectEntityType','predicate']).index.map(label_count.get)
    df_links.to_csv(csv_links, index=False)
    freq_occur_entity()

def freq_occur_entity():
    print("inside freq_occur_entity")
    df_links = pd.read_csv(csv_links)
    for column in ['SubjectEntityType','ObjectEntityType']:
        new_col_name = "Count" + column
        df_links[new_col_name] = df_links[column].map(Counter(list(df_links[column])))
    df_links.to_csv(csv_links, index=False)
    feature_reduction()

def feature_reduction():
    print("Inside feature reduction")
    
    # Load the dataset
    df = pd.read_csv(csv_links)
    
    # Work on a copy to avoid chained assignment issues
    df_filtered = df.iloc[:, 3:].copy()

    # Separate numeric and non-numeric columns
    numeric_columns = df_filtered.select_dtypes(include=[np.number]).columns
    non_numeric_columns = df_filtered.select_dtypes(exclude=[np.number]).columns

    # Replace and cast values for non-numeric columns only
    for feature in non_numeric_columns:
        df_filtered[feature].replace(
            ['1', '0', 'True', 'False', True, False],
            [1, 0, 1, 0, 1, 0],
            inplace=True
        )
        # Attempt conversion to numeric
        df_filtered[feature] = pd.to_numeric(df_filtered[feature], errors='coerce')

    # Drop columns with only one unique value
    for col in numeric_columns:
        if len(df_filtered[col].unique()) == 1:
            print(f"Dropping column with one unique value: {col}")
            df_filtered.drop(columns=[col], inplace=True)

    # Identify highly correlated features
    corr_feature_list = []
    columns = list(df_filtered.columns)
    for i in range(len(columns) - 1):
        for j in range(i + 1, len(columns)):
            try:
                correlation = df_filtered[columns[i]].corr(df_filtered[columns[j]])
                if correlation == 1:
                    print(f"Highly correlated: {columns[i]} and {columns[j]}")
                    corr_feature_list.append(columns[i])
                    corr_feature_list.append(columns[j])
            except Exception as e:
                print(f"Error calculating correlation between {columns[i]} and {columns[j]}: {e}")

    # Remove correlated features
    remove_corr_features(corr_feature_list, df_filtered, df)


def remove_corr_features(corr_feature_list, df_filtered, df):
    print("Correlated Features: ", corr_feature_list)
    
    # Request user input to remove specific features
    features_to_remove = input("Enter features to remove (comma-separated, no spaces): ").split(',')
    features_to_remove = [f for f in features_to_remove if f]  # Remove empty strings
    
    if not features_to_remove:  # If user doesn't input features, skip this step
        print("No features removed. Proceeding to generate binary features.")
    else:
        for feature in features_to_remove:
            if feature in df_filtered.columns:
                df_filtered.drop(columns=[feature], inplace=True)
                print(f"Feature removed: {feature}")
            else:
                print(f"Feature not found: {feature}")
    
    # Generate binary features from the filtered DataFrame
    gen_binary_feature(df_filtered, df)


def gen_binary_feature(df_fileterd, df):
    print("inside binary_features")
    columns = df_fileterd.columns
    for column in columns:
        new_col = []
        new_col_name = "Freq" + column
        for index, row in df_fileterd.iterrows():
            if row[column] > df_fileterd[column].median():
                new_col.append(1)
            else:
                new_col.append(0)
        df[new_col_name] = new_col
    df.to_csv(csv_with_features, index=False)



if __name__ == "__main__":
    construct_df()