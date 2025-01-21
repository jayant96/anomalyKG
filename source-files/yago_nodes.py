from rdflib import Graph, RDF, Literal, RDFS, plugin, OWL, XSD, SKOS, PROV
import csv
import pandas as pd
import numpy as np
from collections import Counter
import spacy
import jellyfish as jf
import json
import validators
import re

# File paths
csv_links = "../results/all_triples.csv"
csv_node_labels = "../results/all_triples.csv"
csv_nodes = "../results/nodes_features.csv"
entity_file = "../results/entities_links.csv"
data_type_json = "../results/data_type_validation.json"
csv_with_features = "../results/features_selected_nodes.csv"

# Load Models
nlp = spacy.load("en_core_sci_sm")

############################ Triple Features ##############################################

def pred_occur():
    print("inside pred_occur")
    df_node_labels = pd.read_csv(csv_node_labels)
    df_node_labels['CountPredOccur'] = df_node_labels['predicate'].map(Counter(list(df_node_labels['predicate'])))
    df_node_labels.to_csv(csv_node_labels, index=False)
    subj_pred_occur()

def subj_pred_occur():
    print("inside subj_pred_occur")
    df_node_labels = pd.read_csv(csv_node_labels)
    total_groups = df_node_labels.groupby(['subject', 'predicate'])
    label_count = {group: len(total_groups.get_group(group)) for group in total_groups.groups}
    df_node_labels['CountPredOccurofSubject'] = df_node_labels.set_index(['subject', 'predicate']).index.map(label_count.get)
    df_node_labels.to_csv(csv_node_labels, index=False)
    dup_triples()

def dup_triples():
    print("inside dup_triples")
    df_node_labels = pd.read_csv(csv_node_labels)
    total_groups = df_node_labels.groupby(['subject', 'predicate', 'object'])
    label_count = {group: len(total_groups.get_group(group)) for group in total_groups.groups}
    df_node_labels['CountDupTriples'] = df_node_labels.set_index(['subject', 'predicate', 'object']).index.map(label_count.get)
    df_node_labels.to_csv(csv_node_labels, index=False)
    cal_haslabel_similarity()

def cal_haslabel_similarity():
    print("inside cal_haslabel_similarity")
    df_node_labels = pd.read_csv(csv_node_labels)
    label_count = {}
    for index, row in df_node_labels.iterrows():
        if row['predicate'] == "has_label":
            entity_split = row['subject'].split("/")[-1]
            underscore_removed = entity_split.replace("_", " ")
            wordnet_removed = underscore_removed.replace('wordnet', "")
            wikicat_removed = wordnet_removed.replace('wikicategory', "")
            subject_final = "".join(filter(lambda x: not x.isdigit(), wikicat_removed)).strip()
            similarity = jf.jaro_distance(subject_final, row['object'])
            label_count[(row['subject'], row['object'])] = round(similarity, 2)
        else:
            label_count[(row['subject'], row['object'])] = "na"
    df_node_labels['SimSubjectObject'] = df_node_labels.set_index(['subject', 'object']).index.map(label_count.get)
    df_node_labels.to_csv(csv_node_labels, index=False)
    tot_literals()

############################ Entity Recognition ##########################################

def entity_recognition():
    print("Running Entity Recognition for FB15K")

    # Use a general-purpose spaCy model
    nlp = spacy.load("en_core_web_sm")

    # Read the CSV containing your triples or statements
    df_node_labels = pd.read_csv(csv_node_labels)

    # Collect all unique entities from subject and object columns
    entities = set(df_node_labels['subject']).union(df_node_labels['object'])
    dict_entity_type = {}
    total_entities = len(entities)
    print(f"Total entities to process: {total_entities}")

    with open(entity_file, "w", newline='', encoding="utf-8") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["entity", "entity_type"])
        count = 0

        for entity in entities:
            count += 1
            print(f"Processing entity {count}/{total_entities}")

            # Example cleanup: if your FB15K IDs are like /m/02mmwk,
            # just take the last part and remove underscores or digits if needed.
            entity_split = entity.split("/")[-1]      # e.g. "02mmwk"
            underscore_removed = entity_split.replace("_", " ")

            # The following lines (removing 'wordnet', 'wikicategory', digits) were YAGO-specific,
            # so you can remove or keep them if you still have those patterns in your data.
            # For FB15K, you might just do:
            entity_final = underscore_removed.strip()

            # Process with spaCy NER
            doc = nlp(entity_final)
            entity_types = [ent.label_ for ent in doc.ents]

            # Fallback if spaCy doesn't recognize anything
            if not entity_types:
                entity_types = ["FREEBASE_ENTITY"]  # or "UNKNOWN"

            dict_entity_type[entity] = set(entity_types)
            writer.writerow([entity, set(entity_types)])

    # Update the DataFrame with recognized types
    df_node_labels['SubjectEntityType'] = df_node_labels['subject'].map(dict_entity_type)
    df_node_labels['ObjectEntityType'] = df_node_labels['object'].map(dict_entity_type)

    # Save the changes
    df_node_labels.to_csv(csv_node_labels, index=False)

    print("Entity Recognition Complete")
    # Call the next step in your pipeline
    pred_entity_type_occur()


def pred_entity_type_occur():
    print("inside pred_entity_type_occur")
    df_node_labels = pd.read_csv(csv_node_labels)
    total_groups = df_node_labels.groupby(['SubjectEntityType', 'ObjectEntityType', 'predicate'])
    label_count = {}
    for group in total_groups.groups:
        try:
            label_count[group] = len(total_groups.get_group(group))
        except Exception as e:
            print(f"Error processing group {group}: {e}")
            continue
    df_node_labels['CountPredOccurEntityType'] = df_node_labels.set_index(['SubjectEntityType', 'ObjectEntityType', 'predicate']).index.map(label_count.get)
    df_node_labels.to_csv(csv_node_labels, index=False)
    print("Finished calculating predicate occurrences with entity types.")
    find_com_rare_entity_type()


############################ Node-Level Features ##########################################

def tot_literals():
    print("inside tot_literals")
    df_node_labels = pd.read_csv(csv_node_labels)
    total_groups = df_node_labels.groupby(['subject'])
    label_count = {group: len(total_groups.get_group(group)) for group in total_groups.groups}
    data = {'node': list(label_count.keys()), 'CountLiterals': list(label_count.values())}
    df_nodes = pd.DataFrame.from_dict(data)
    df_nodes.to_csv(csv_nodes, index=False)
    count_dif_literal_types()

def count_dif_literal_types():
    print("inside count_dif_literal_types")
    df_node_labels = pd.read_csv(csv_node_labels)
    df_nodes = pd.read_csv(csv_nodes)
    total_groups = df_node_labels.groupby(['subject'])
    label_count = {group: len(total_groups.get_group(group)['predicate'].unique()) for group in total_groups.groups}
    df_nodes['CountDifLiteralTypes'] = df_nodes.set_index(['node']).index.map(label_count.get)
    df_nodes.to_csv(csv_nodes, index=False)
    count_isa_triples()

def count_isa_triples():
    print("inside count_isa_triples")
    df_node_labels = pd.read_csv(csv_node_labels)
    df_nodes = pd.read_csv(csv_nodes)
    total_groups = df_node_labels.groupby(['subject'])
    label_count = {group: len(total_groups.get_group(group)[total_groups.get_group(group)['predicate'] == 'isa']) for group in total_groups.groups}
    df_nodes['CountIsaPredicate'] = df_nodes.set_index(['node']).index.map(label_count.get)
    df_nodes.to_csv(csv_nodes, index=False)
    count_haslabel_triples()

def count_haslabel_triples():
#count different types of predicates an entity has got where the object is a literal
    print("inside count_haslabel_triples")
    df_node_labels = pd.read_csv(csv_node_labels)
    df_nodes = pd.read_csv(csv_nodes)
    total_groups = df_node_labels.groupby(['subject'])
    label_count = {}
    for group in total_groups.groups:
        fetched_group = total_groups.get_group(group)
        print(fetched_group)
        count_dif_literals = len(fetched_group[fetched_group['predicate']=='has_label'])
        label_count[group] = count_dif_literals
    df_nodes['CountHaslabelPredicate'] = df_nodes.set_index(['node']).index.map(label_count.get)
    df_nodes.to_csv(csv_nodes, index=False)
    count_subclassof_triples()


def count_subclassof_triples():
#count different types of predicates an entity has got where the object is a literal
    print("inside count_subclassof_triples")
    df_node_labels = pd.read_csv(csv_node_labels)
    df_nodes = pd.read_csv(csv_nodes)
    total_groups = df_node_labels.groupby(['subject'])
    label_count = {}
    for group in total_groups.groups:
        fetched_group = total_groups.get_group(group)
        print(fetched_group)
        count_dif_literals = len(fetched_group[fetched_group['predicate']=='subClassOf'])
        label_count[group] = count_dif_literals
    df_nodes['CountSubclassofPredicate'] = df_nodes.set_index(['node']).index.map(label_count.get)
    df_nodes.to_csv(csv_nodes, index=False)
    count_subpropertyof_triples()

def count_subpropertyof_triples():
#count different types of predicates an entity has got where the object is a literal
    print("inside count_subpropertyof_triples")
    df_node_labels = pd.read_csv(csv_node_labels)
    df_nodes = pd.read_csv(csv_nodes)
    total_groups = df_node_labels.groupby(['subject'])
    label_count = {}
    for group in total_groups.groups:
        fetched_group = total_groups.get_group(group)
        print(fetched_group)
        count_dif_literals = len(fetched_group[fetched_group['predicate']=='subPropertyOf'])
        label_count[group] = count_dif_literals
    df_nodes['CountSubpropofPredicate'] = df_nodes.set_index(['node']).index.map(label_count.get)
    df_nodes.to_csv(csv_nodes, index=False)
    count_high_sim_labels()

def count_high_sim_labels():
#count different types of predicates an entity has got where the object is a literal
    print("inside count_high_sim_labels")
    df_node_labels = pd.read_csv(csv_node_labels)
    df_nodes = pd.read_csv(csv_nodes)
    total_groups = df_node_labels.groupby(['subject'])
    label_count = {}
    for group in total_groups.groups:
        fetched_group = total_groups.get_group(group)
        count = 0
        for index, row in fetched_group.iterrows():
            if row['SimSubjectObject'] != 'na' and float(row['SimSubjectObject']) >=0.5:
                count+=1
        label_count[group] = count
    df_nodes['CountHighSimLabels'] = df_nodes.set_index(['node']).index.map(label_count.get)
    df_nodes.to_csv(csv_nodes, index=False)
    count_low_sim_labels()

def count_low_sim_labels():
#count different types of predicates an entity has got where the object is a literal
    print("inside count_low_sim_labels")
    df_node_labels = pd.read_csv(csv_node_labels)
    df_nodes = pd.read_csv(csv_nodes)
    total_groups = df_node_labels.groupby(['subject'])
    label_count = {}
    for group in total_groups.groups:
        fetched_group = total_groups.get_group(group)
        count = 0
        for index, row in fetched_group.iterrows():
            if row['SimSubjectObject'] != 'na' and float(row['SimSubjectObject']) <0.5:
                count+=1
        label_count[group] = count
    df_nodes['CountLowSimLabels'] = df_nodes.set_index(['node']).index.map(label_count.get)
    df_nodes.to_csv(csv_nodes, index=False)
    tot_outgoing_links()

def tot_outgoing_links():
    print("inside tot_incoming_links")
    df_node_labels = pd.read_csv(csv_node_labels)
    df_nodes = pd.read_csv(csv_nodes)
    df_links = pd.read_csv(csv_links)
    groups_in_node_labels = df_node_labels.groupby(['subject'])
    groups_in_links = df_links.groupby(['subject'])
    label_count = {}
    for group in df_nodes['node']:
        try:
            fetched_node_label_groups = len(groups_in_node_labels.get_group(group))
        except:
            fetched_node_label_groups = 0
        try:
            fetched_link_groups = len(groups_in_links.get_group(group))
        except:
            fetched_link_groups = 0
        label_count[group] = fetched_node_label_groups + fetched_link_groups
    df_nodes['OutDegree'] = df_nodes.set_index(['node']).index.map(label_count.get)
    df_nodes.to_csv(csv_nodes, index=False)
    tot_incoming_links()

def tot_incoming_links():
    print("inside tot_outgoing_links")
    df_nodes = pd.read_csv(csv_nodes)
    df_links = pd.read_csv(csv_links)
    groups_in_links = df_links.groupby(['object'])
    label_count = {}
    for group in df_nodes['node']:
        try:
            fetched_link_groups = len(groups_in_links.get_group(group))
        except:
            fetched_link_groups = 0
        label_count[group] = fetched_link_groups
    df_nodes['InDegree'] = df_nodes.set_index(['node']).index.map(label_count.get)
    df_nodes.to_csv(csv_nodes, index=False)
    validate_literal_data_type()

def validate_literal_data_type():
    print("inside validate_literal_data_type")
    df_node_labels = pd.read_csv(csv_node_labels)

    # Load data type configuration
    try:
        with open(data_type_json, 'r') as json_file:
            data_type_dict = json.load(json_file)
    except FileNotFoundError:
        print(f"Error: File {data_type_json} not found. Skipping literal validation.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not parse {data_type_json}. Ensure it is a valid JSON file.")
        return

    print(f"Loaded data type configuration: {data_type_dict}")

    validity_score = []
    for index, row in df_node_labels.iterrows():
        pred_extracted = row['predicate']
        if validators.url(pred_extracted):
            pred_extracted = row['predicate'].split("/")[-1]

        validity = "na"
        for key in data_type_dict.keys():
            if key in pred_extracted.lower():
                data_type = data_type_dict[key]
                try:
                    if data_type == "url":
                        validity = is_url(row['object'])
                        break
                    if data_type == "date":
                        validity = is_date(row['object'])
                        break
                    if data_type == 'integer':
                        validity = row['object'].isnumeric()
                        break
                    if data_type == 'time':
                        validity = re.match(r'\d{2}:\d{2}:\d{2}', row['object']) is not None
                        break
                    if data_type == 'string':
                        validity = (
                            not row['object'].isnumeric()
                            and row['object'] != ""
                            and not validators.url(row['object'])
                            and not is_date(row['object'])
                        )
                        break
                except Exception as e:
                    print(f"Error validating literal {row['object']} for predicate {row['predicate']}: {e}")
                    validity = False
        validity_score.append(validity)

    df_node_labels['ValidityOfLiteral'] = validity_score
    df_node_labels.to_csv(csv_node_labels, index=False)
    print("Finished validating literals.")
    count_occur_dup_triples()


def find_com_rare_entity_type():
# counts the no. of times a predicate occurs within the entity's group
    print("inside find_com_rare_entity_type")
    df_node_links = pd.read_csv(csv_node_labels)
    df_nodes = pd.read_csv(csv_nodes)
    total_groups = df_node_links.groupby(['subject'])
    entity_count_node_max, entity_count_node_min = {}, {}
    for group in total_groups.groups:
        entity_count = {}
        sub_group = total_groups.get_group(group).groupby(['SubjectEntityType','ObjectEntityType'])
        for entity_group in sub_group.groups:
            entity_count[entity_group] = len(sub_group.get_group(entity_group))
        key_max = max(entity_count.keys(), key=(lambda k: entity_count[k]))
        key_min = min(entity_count.keys(), key=(lambda k: entity_count[k]))
        entity_count_node_max[group] = [key_max]
        entity_count_node_min[group] = [key_min]
    df_nodes['CommonPredType'] = df_nodes.set_index(['node']).index.map(entity_count_node_max.get)
    df_nodes['RarePredType'] = df_nodes.set_index(['node']).index.map(entity_count_node_min.get)
    df_nodes.to_csv(csv_nodes, index=False)

def count_occur_dup_triples():
# counts the no. of duplicate triples a node has got
    print("inside count_occur_dup_triples")
    df_node_labels = pd.read_csv(csv_node_labels)
    df_nodes = pd.read_csv(csv_nodes)
    total_groups = df_node_labels.groupby(['subject'])
    label_count = {}
    for group in total_groups.groups:
        count=0
        for index, row in total_groups.get_group(group).iterrows():
            if row['CountDupTriples'] > 1:
                count+=row['CountDupTriples']
        label_count[group] = count
    df_nodes['CountDupTriples'] = df_nodes.set_index(['node']).index.map(label_count.get)
    df_nodes.to_csv(csv_nodes, index=False)
    count_invalid_literals()

def count_invalid_literals():
    print("inside count_invalid_literals")
    df_node_labels = pd.read_csv(csv_node_labels)
    df_nodes = pd.read_csv(csv_nodes)
    total_groups = df_node_labels.groupby(['subject'])
    label_count = {}
    for group in total_groups.groups:
        count=0
        for index, row in total_groups.get_group(group).iterrows():
            if row['ValidityOfLiteral'] == False:
                count+=1
        label_count[group] = count
    df_nodes['CountInvalidTriples'] = df_nodes.set_index(['node']).index.map(label_count.get)
    df_nodes.to_csv(csv_nodes, index=False)
    entity_recognition()

########################################Special Functions###################################
def is_date(string, fuzzy=False):
    """
    Return whether the string can be interpreted as a date.
    :param string: str, string to check for date
    :param fuzzy: bool, ignore unknown tokens in string if True
    """
    from dateutil.parser import parse
    try:
        parse(string, fuzzy=fuzzy)
        return True

    except ValueError:
        return False

def is_url(url):
#check for valid URL
    if not validators.url(url):
        return False
    else:
        return True

def feature_reduction():
    print("Inside feature reduction")
    df = pd.read_csv(csv_node_labels)
    df_fileterd = df.iloc[:, 3:]  # Start processing columns from the 4th column onwards

    # Ensure only numeric data is processed
    for feature in df_fileterd.columns:
        if not np.issubdtype(df_fileterd[feature].dtype, np.number):
            try:
                # Attempt to convert to numeric (ignore errors)
                df_fileterd[feature] = pd.to_numeric(df_fileterd[feature], errors='coerce')
            except Exception as e:
                print(f"Skipping column {feature} due to conversion error: {e}")
                continue

    # Drop columns with all NaN values (after attempting numeric conversion)
    df_fileterd = df_fileterd.dropna(axis=1, how='all')

    # Drop columns with only one unique value
    for col in df_fileterd.columns:
        count_unique = len(df_fileterd[col].dropna().unique())
        if count_unique == 1:
            print(f"Dropping column with one unique value: {col}")
            df_fileterd.drop(col, inplace=True, axis=1)

    # Identify and handle highly correlated features
    columns = list(df_fileterd.columns)
    corr_feature_list = []
    for i in range(len(columns) - 1):
        for j in range(i + 1, len(columns)):
            try:
                correlation = df_fileterd[columns[i]].corr(df_fileterd[columns[j]])
                if correlation == 1:
                    print(f"Highly correlated features: {columns[i]} and {columns[j]}")
                    corr_feature_list.append(columns[i])
                    corr_feature_list.append(columns[j])
            except Exception as e:
                print(f"Error calculating correlation between {columns[i]} and {columns[j]}: {e}")

    remove_corr_features(corr_feature_list, df_fileterd, df)

def remove_corr_features(corr_feature_list, df_fileterd, df):
    print("Correlated Features: ", corr_feature_list)

    # Prompt user for feature selection
    features_to_remove = input(
        f"Enter the features to remove (comma-separated, no spaces): "
    ).strip().split(',')

    if not features_to_remove or features_to_remove == ['']:
        print("No features removed. Proceeding with all features.")
    else:
        for feature in features_to_remove:
            feature = feature.strip()  # Remove any extra whitespace
            if feature in df_fileterd.columns:
                df_fileterd.drop(columns=[feature], inplace=True)
                print(f"Removed feature: {feature}")
            else:
                print(f"Feature not found: {feature}. Skipping.")

    gen_binary_feature(df_fileterd, df)

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

############################ Main Execution ###############################################

if __name__ == "__main__":
    print("Starting node-level feature generation...")
    pred_occur()
    feature_reduction()
    

