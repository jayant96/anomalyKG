import numpy as py
import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import parameters as pm

def get_params(dataset):
    pm.params(dataset)
    global feature_file, dataframemain, svm_output_pkl, initial_attribute_count
    feature_file = pm.params.feature_file
    dataframemain = pd.read_csv(feature_file)
    svm_output_pkl = pm.params.svm_output_pkl
    initial_attribute_count = pm.params.initial_attribute_count
    get_abnormal_counts()

def get_abnormal_counts():
    """
    Detect anomalous triples using One-Class SVM with improved preprocessing,
    dynamic thresholding, and better scoring.
    """
    print("Generating abnormal counts and scores")
    list_of_date_postcodes = list(dataframemain.index)
    new_dataframe = pd.DataFrame(list_of_date_postcodes)
    
    # Preprocess the data
    yagodf = dataframemain.iloc[:, initial_attribute_count:]
    yagodf = preprocess_data(yagodf)
    
    # Dynamic nu calculation
    records_count = len(yagodf.index)
    desired_outlier_fraction = 0.1 # Adjust based on your dataset
    nu = desired_outlier_fraction  # Can be dynamically tuned per dataset
    
    # Iterate over kernels
    for kernel in ['rbf', 'sigmoid', 'linear', 'poly']:
        all_dic, binary_outlier, neg_dic, pos_dic, datanormalized = {}, {}, {}, {}, []
        
        # Train the One-Class SVM
        clf = OneClassSVM(kernel=kernel, nu=nu).fit(yagodf)
        detect_binary_outliers = clf.predict(yagodf)
        detect_score_outliers = clf.decision_function(yagodf)
        
        # Add outputs to the new dataframe
        new_dataframe[kernel + "_binary_output"] = detect_binary_outliers
        new_dataframe[kernel + "_score_output"] = py.round(detect_score_outliers, 5)
        
        # Normalize scores
        normalized_scores = normalize_scores(detect_score_outliers, detect_binary_outliers)
        new_dataframe[kernel + "_score_output_normalized"] = normalized_scores

    # Process abnormal pickups
    count_abnormal_pickups(new_dataframe)

def normalize_scores(scores, binary_outliers):
    """
    Normalize the SVM decision scores between 0 and 1 for both positive and negative cases.
    """
    neg_scores = [score for score, outlier in zip(scores, binary_outliers) if outlier == -1]
    pos_scores = [score for score, outlier in zip(scores, binary_outliers) if outlier != -1]

    min_val_neg, max_val_neg = min(neg_scores, default=0), max(neg_scores, default=0)
    min_val_pos, max_val_pos = min(pos_scores, default=0), max(pos_scores, default=0)

    normalized_scores = []
    for score, outlier in zip(scores, binary_outliers):
        if outlier == -1:  # Normalize negatives
            normalized = (((score - min_val_neg) / (max_val_neg - min_val_neg + 1e-8)) - 1)
        else:  # Normalize positives
            normalized = (score - min_val_pos) / (max_val_pos - min_val_pos + 1e-8)
        normalized_scores.append(round(normalized, 5))
    
    return normalized_scores

def count_abnormal_pickups(dataframemain):
    """
    Count how many kernels mark a row as an anomaly.
    """
    count_list = []
    for index, row in dataframemain.iterrows():
        count = sum(1 for col in [
            row['sigmoid_binary_output'], 
            row['rbf_binary_output'], 
            row['linear_binary_output'], 
            row['poly_binary_output']] if col == -1)
        count_list.append(count)
    dataframemain['count_of_abnormal_pickups'] = count_list
    sum_abnormal_score(dataframemain)

def sum_abnormal_score(dataframemain):
    """
    Sum normalized scores across all kernels for each row.
    """
    count_list = []
    for index, row in dataframemain.iterrows():
        count = sum([
            row['sigmoid_score_output_normalized'], 
            row['rbf_score_output_normalized'],
            row['linear_score_output_normalized'], 
            row['poly_score_output_normalized']])
        count_list.append(count)
    dataframemain['total_score_abnormal_pickups'] = count_list
    calculate_average_score(dataframemain)

def calculate_average_score(dataframemain):
    """
    Calculate the average anomaly score and determine final anomaly status.
    """
    count_list = []
    for index, row in dataframemain.iterrows():
        si = row['total_score_abnormal_pickups'] / 4
        count_list.append(round(si, 3))
    dataframemain['average_score'] = count_list
    dataframemain = dataframemain.sort_values(by=["average_score"], ascending=True)

    # Final anomaly label based on average score
    list_adjusted_svm = [-1 if row['average_score'] <= 0 else 1 for _, row in dataframemain.iterrows()]
    dataframemain['adjusted_svm'] = list_adjusted_svm
    reconstruct_dataframe(dataframemain)

def reconstruct_dataframe(new_dataframemain):
    """
    Map the SVM results back to the original dataframe.
    """
    global dataframemain
    dict_svm_output = dict(zip(new_dataframemain.index, new_dataframemain['adjusted_svm']))
    dict_weighted_score = dict(zip(new_dataframemain.index, new_dataframemain['average_score']))

    old_dataframemain = dataframemain.copy()
    old_dataframemain['svm_binary_output'] = old_dataframemain.index.map(dict_svm_output)
    old_dataframemain['average_score'] = old_dataframemain.index.map(dict_weighted_score)
    old_dataframemain = old_dataframemain.sort_values(by=["average_score"], ascending=True)
    old_dataframemain.to_pickle(svm_output_pkl)
    print("SVM learning completed...")

def preprocess_data(dataframe):
    """
    Preprocess the dataframe to handle categorical and numerical data.
    """
    categorical_columns = ['SubjectEntityType', 'ObjectEntityType']
    dataframe[categorical_columns] = dataframe[categorical_columns].replace('UNKNOWN', None)
    dataframe = pd.get_dummies(dataframe, columns=categorical_columns, dummy_na=True)
    dataframe = dataframe.fillna(0)  # Replace missing values
    scaler = StandardScaler()
    dataframe_scaled = scaler.fit_transform(dataframe)
    return pd.DataFrame(dataframe_scaled, index=dataframe.index)

if __name__ == "__main__":
    get_params(3)
