# Generate train-test-files.py: generate indexes for testing and training
#

import sys
import csv
import math
import random

import numpy as np
# from keras.utils import to_categorical
from tensorflow.keras.utils import to_categorical
# from tensorflow.python.keras.utils import to_categorical
from imblearn.over_sampling import SMOTE
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout
from keras.layers import GlobalAveragePooling2D
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import StratifiedKFold

from dataset_loader import DatasetLoader
from dataset_loaderv import DatasetLoaderv

import pandas as pd

EMOTION_NUMBERS = [0, 1, 2, 3, 4, 5, 6]



def load_dataset():
    DATASET_PATH = 'datasets/Original/MFCC'
    dataset_loader = DatasetLoader(DATASET_PATH)
    mfcc_features, y = dataset_loader.get_dataset()

    DATASET_PATH = 'datasets/Original/Prosody'
    dataset_loader = DatasetLoader(DATASET_PATH)
    prosody_features, y = dataset_loader.get_dataset()

    # print('len_mfcc_features:',len(mfcc_features))
    # print('len_prosody_features:', len(prosody_features))
    new_dataset = []
    for index in range(0, len(mfcc_features)):
        new_instance = []
        for row_index in range(0, len(mfcc_features[index])):

            # print('index:', index, ' row_index:', row_index)
            # print('mfcc_features:', mfcc_features[index][row_index])
            # print('prosody_features:', prosody_features[index][row_index])
            new_row = np.concatenate(
                (mfcc_features[index][row_index],
                 prosody_features[index][row_index]),
                axis=None
            )
            new_instance.append(new_row)
        new_dataset.append(new_instance)

    X = new_dataset

    DATASET_PATH = 'datasets/Original/Chroma'
    dataset_loader = DatasetLoader(DATASET_PATH)
    chroma_features, y = dataset_loader.get_dataset()

    new_dataset = []
    for index in range(0, len(chroma_features)):
        new_instance = []
        for row_index in range(0, len(chroma_features[index])):
            new_row = np.concatenate(
                (X[index][row_index],
                 chroma_features[index][row_index]),
                axis=None
            )
            new_instance.append(new_row)
        new_dataset.append(new_instance)
        ### incluido gravacao Omar
        # new_instance_array = np.asarray(new_instance)
        # with open('badem.csv', 'a', newline='') as file:
        #     writer = csv.writer(file, delimiter=';')
        #     writer.writerow(new_instance_array)
        ### fim inclusao gravacao Omar
    X = np.asarray(new_dataset)
    # print ('dataset concatenado:', X[0:3])
    return X, y


def load_datasetv():
    DATASET_PATH = 'datasetsv/Original/MFCC'
    dataset_loaderv = DatasetLoaderv(DATASET_PATH)
    mfcc_features, y = dataset_loaderv.get_dataset()
    DATASET_PATH = 'datasetsv/Original/Prosody'
    dataset_loaderv = DatasetLoaderv(DATASET_PATH)
    prosody_features, y = dataset_loaderv.get_dataset()

    new_dataset = []
    for index in range(0, len(mfcc_features)):
        new_instance = []
        for row_index in range(0, len(mfcc_features[index])):

            # print('index:', index, ' row_index:', row_index)
            # print('mfcc_features:', mfcc_features[index][row_index])
            # print('prosody_features:', prosody_features[index][row_index])
            new_row = np.concatenate(
                (mfcc_features[index][row_index],
                 prosody_features[index][row_index]),
                axis=None
            )
            new_instance.append(new_row)
        new_dataset.append(new_instance)

    X = new_dataset

    DATASET_PATH = 'datasetsv/Original/Chroma'
    dataset_loaderv = DatasetLoaderv(DATASET_PATH)
    chroma_features, y = dataset_loaderv.get_dataset()
    new_dataset = []
    for index in range(0, len(chroma_features)):
        new_instance = []
        for row_index in range(0, len(chroma_features[index])):
            new_row = np.concatenate(
                (X[index][row_index],
                 chroma_features[index][row_index]),
                axis=None
            )
            new_instance.append(new_row)
        new_dataset.append(new_instance)

    X = np.asarray(new_dataset)
    return X, y



print ("sys.argv-str:", str(sys.argv[0:]))
tipo = sys.argv[1]
print("Tipo dataset: TRTE-VERBO(1), TRTE-BADEM(2), TRTE-BADEM-VERBO(3), TRTE-VERBO-BADEM(4), TRTE-ALL(5):", tipo)

if tipo == "1": data_set = "TRTE-VERBO"
elif tipo == "2": data_set = "TRTE-BADEM"
elif tipo == "3": data_set = "TRTE-BADEM-VERBO"
elif tipo == "4": data_set = "TRTE-VERBO-BADEM"
elif tipo == "5": data_set = "TRTE-ALL"
else: print ("tipo deve ser de 1 a 5")

out1 = "output1-" + data_set + ".csv"
out2 = "output2-" + data_set + ".csv"
out3 = "output3-" + data_set + ".csv"

print("data_set1:", out1)
print("data_set2:", out2)
print("data_set3:", out3)

if data_set == "TRTE-BADEM":
    X_dados, y_dados = load_dataset()
elif data_set == "TRTE-VERBO":
    X_dados, y_dados = load_datasetv()
elif data_set == "TRTE-VERBO-BADEM":
    X_dadostr, y_dadostr = load_datasetv()
    X_dadoste, y_dadoste = load_dataset()
elif data_set == "TRTE-BADEM-VERBO":
    X_dadoste, y_dadoste = load_datasetv()
    X_dadostr, y_dadostr = load_dataset()
elif data_set == "TRTE-ALL":
    X_dados, y_dados = load_dataset()
    X_dadosv, y_dadosv = load_datasetv()
    out_datasety = "datasety-" + data_set + ".csv"
    out_datasetx = "datasetx-" + data_set + ".csv"
    X_dados = np.append(X_dados, X_dadosv, axis=0)
    y_dados = np.append(y_dados, y_dadosv, axis=0)
    X_dados = np.asarray(X_dados)
    y_dados = np.asarray(y_dados)

else:
    print ("Invalido dataset especificado.")

if data_set in ("TRTE-BADEM", "TRTE-VERBO", "TRTE-ALL"):
    strat_kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    y_dados0 = [int(inst_y[0]) for inst_y in y_dados]
    kfold = 0
    for train_idx, test_idx in strat_kfold.split(X_dados, y_dados0):
        outtr = data_set + "-train-idx-" + str(kfold) + ".csv"
        with open(outtr, 'w', newline='') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow(train_idx)
        outte = data_set + "-test-idx-" + str(kfold) + ".csv"
        with open(outte, 'w', newline='') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow(test_idx)
        train_file = pd.read_csv(outtr, delimiter=';', header=None)
        train_x = train_file.to_numpy(dtype=int,copy=True).ravel()
        test_file = pd.read_csv(outte, delimiter=';',header=None)
        test_x = test_file.to_numpy(dtype=int,copy=True).ravel()
        kfold = kfold + 1

