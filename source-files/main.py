import os
import time
# import requests
# import py7zr
# import zipfile
# import tarfile

if not os.path.isdir("../results"):
    os.mkdir("../results")

if not os.path.isdir("../assets"):
    os.mkdir("../assets")

# print("Downloading KG")
# response = requests.get('https://yago-knowledge.org/data/yago1/yago-1.0.0-turtle.7z')
# with open("../assets/yago-1.0-turtle.7z", 'wb') as yago:
#     yago.write(response.content)

# print("Extracting the downloaded KG")
# with py7zr.SevenZipFile('../assets/yago-1.0-turtle.7z', mode='r') as extyago:
#     extyago.extractall(path='../assets')

# print("Downloading Stanford NER 4.2.0")
# response = requests.get('https://nlp.stanford.edu/software/stanford-ner-4.2.0.zip')
# with open("../assets/stanford-ner-4.2.0.zip", 'wb') as stanford:
#     stanford.write(response.content)

# print("Extracting Stanford NER 4.2.0")
# with zipfile.ZipFile("../assets/stanford-ner-4.2.0.zip", 'w') as stan:
#     stan.extractall(path='../assets')

# print("Downloading en_core_web_sm-3.0.0")
# response = requests.get("https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.0.0/en_core_web_sm-3.0.0.tar.gz")
# with open("../assets/en_core_web_sm-3.0.0.tar.gz", 'wb') as encore:
#     encore.write(response.content)

# print("Extracting en_core_web_sm-3.0.0")
# fname = "../assets/en_core_web_sm-3.0.0.tar.gz"
# tar = tarfile.open(fname, "r:gz")
# tar.extractall(path='../assets')

import yago_links as yl
import yago_nodes as nl
import svm_training as svm
import pattern_generation as pg

def generate_features():
    print("-----------------------------------")
    print("Generating features for UMLS 0.01 dataset")
    dataset=3
    start_time = time.time()
    yl.construct_df()
    execution_time = time.time() - start_time
    print('Total runtime taken for feature generation: %.6f sec' % (execution_time))
    learn_one_class_svm(dataset)
    visualization(dataset)

    # dataset = 2
    # start_time = time.time()
    # nl.pred_occur()
    # execution_time = time.time() - start_time
    # print('Total runtime taken for feature generation: %.6f sec' % (execution_time))
    # learn_one_class_svm(dataset)
    # visualization(dataset)

def learn_one_class_svm(dataset):
    print("-----------------------------------")
    print("Started learning one-class SVM")
    start_time = time.time()
    svm.get_params(dataset)
    execution_time = time.time() - start_time
    print('Total runtime taken for SVM training: %.6f sec' % (execution_time))

def visualization(dataset):
    print("-----------------------------------")
    print("Started visualizing the results")
    start_time = time.time()
    pg.get_params(dataset)
    execution_time = time.time() - start_time
    print('Total runtime taken for visualization: %.6f sec' % (execution_time))

generate_features()



