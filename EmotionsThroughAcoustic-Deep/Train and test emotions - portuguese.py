# Train and test emotions - portuguese.py
#

import sys
import csv
import math
import random

import numpy as np
from tensorflow.keras.utils import to_categorical
from imblearn.over_sampling import SMOTE
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout
from keras.layers import GlobalAveragePooling2D
from keras.models import Sequential
# import utilizado no desktop pessoal
from keras.preprocessing.sequence import pad_sequences
# import utilizado no servidor do Lapis
# from keras.utils import pad_sequences

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

from dataset_loader import DatasetLoader
from dataset_loaderv import DatasetLoaderv

import pandas as pd

EMOTION_NUMBERS = [0, 1, 2, 3, 4, 5, 6]
# Parâmetros de treinamento para cada emoção:
#            '0' : 'neu'
#            '1' : 'des'
#            '2' : 'med'
#            '3' : 'ale'
#            '4' : 'rai'
#            '5' : 'sur'
#            '6' : 'tri' 
emoparm = []
emoparm.append([128, 0.4, 256, 0.6, 'RMSprop'])
emoparm.append([256, 0.6, 64, 0.4, 'RMSprop'])
emoparm.append([128, 0.6, 256, 0.6, 'Adam'])
emoparm.append([128, 0.6, 256, 0.6, 'RMSprop'])
emoparm.append([256, 0.6, 64, 0.4, 'RMSprop'])
emoparm.append([32, 0.4, 256, 0.6, 'Adam'])
emoparm.append([64, 0.2, 64, 0.4, 'SGD'])


def add_padding(X, y):

    X = np.asarray(X)
    y = np.asarray(y)
    max_len = len(X[0])
    for row in X:
        if len(row) > max_len:
            max_len = len(row)

    X = pad_sequences(X, maxlen=max_len, padding='post', dtype='float64')
    return X, y


def add_padding2(X, y, X_test, y_test):
    X = np.asarray(X)
    y = np.asarray(y)
    max_len = len(X[0])
    for row in X:
        if len(row) > max_len:
            max_len = len(row)
    sum_len = 0
    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)
    X = pad_sequences(X, maxlen=max_len, padding='post', dtype='float64')
    X_test = pad_sequences(X_test, maxlen=max_len, padding='post', dtype='float64')
    return X, y, X_test, y_test


def load_dataset():
    DATASET_PATH = 'datasets/Original/MFCC'
    dataset_loader = DatasetLoader(DATASET_PATH)
    mfcc_features, y = dataset_loader.get_dataset()

    DATASET_PATH = 'datasets/Original/Prosody'
    dataset_loader = DatasetLoader(DATASET_PATH)
    prosody_features, y = dataset_loader.get_dataset()

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
    X = np.asarray(new_dataset, dtype=object)
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

    X = np.asarray(new_dataset, dtype=object)
    return X, y


def apply_smote(X, y):
    shape_0 = X.shape[0]
    shape_1 = X.shape[1]
    shape_2 = X.shape[2]
    X = X.reshape(shape_0, shape_1 * shape_2)

    # Apply SMOTE
    smt = SMOTE()
    try:
        X, y = smt.fit_sample(X, y)
    except:
        pass

    # Reshaping back to original shape dimensions 1 and 2
    X = X.reshape(X.shape[0], shape_1, shape_2)
    
    return X, y


def wanted_emotion_indexes(w_emotion, y_all):
    # selection of indexes of desired emotion
    indexes = []
    for i in range(0, len(y_all)):
        if y_all[i] == w_emotion:
            indexes.append(i)
    return indexes

def run(wr_emotion, run_name, X_all, y_dados_d, X_test_d, y_test_d):
    y_all = y_dados_d.copy()
    X_test = X_test_d
    y_test = y_test_d.copy()
    y_list = []
    # Create list of indexes for each emotion
    for emotion in EMOTION_NUMBERS:
        current_emotion_indexes = wanted_emotion_indexes(emotion, y_all)
        y_list.append(current_emotion_indexes)

    # Create a list of wanted indexes containing all indexes for the desired emotion.
    # That number will be the first half of the list, the other half will contain
    # A slice of random indexes from the other emotions. This will result in a wanted list
    # with half of indexes of the desired emotion and other half with other emotions
    indexes_wanted = y_list[wr_emotion]

    half_dataset_number = len(y_list[wr_emotion])

    other_emotions_number = math.ceil(half_dataset_number/(len(EMOTION_NUMBERS)-1))

    for emotion in EMOTION_NUMBERS:
        if emotion != wr_emotion:
            indexes_wanted = indexes_wanted + random.sample(y_list[emotion], other_emotions_number)

    # Adjust the labels as being the desired emotion or not
    y_all = binary_emotion_label(wr_emotion, y_all)
    y_test = binary_emotion_label(wr_emotion, y_test)

    X = np.take(X_all, indexes_wanted, axis=0)

    y = np.take(y_all, indexes_wanted, axis=0)
    X = np.asarray(X)

    X_test = np.asarray(X_test)
    X, y, X_test, y_test = add_padding2(X, y, X_test, y_test)

    X_train, X_test_val, y_train, y_test_val = train_test_split(X, y,
                                                                stratify=y,
                                                                test_size=0.3)
    X_train, y_train = apply_smote(X, y)
    y_train = to_categorical(y_train)
    y_test_val = to_categorical(y_test_val)
    y_labels = y_test.copy()
    y_test = to_categorical(y_test)
    num_rows = X_train[0].shape[0]
    num_columns = X_train[0].shape[1]
    num_rowstv = X_test_val[0].shape[0]
    num_columnstv = X_test_val[0].shape[1]
    num_rowst = X_test[0].shape[0]
    num_columnst = X_test[0].shape[1]
    num_channels = 1

    X_train = X_train.reshape(X_train.shape[0], num_rows, num_columns, num_channels)
    X_test = X_test.reshape(X_test.shape[0], num_rowst, num_columns, num_channels)
    X_test_val = X_test_val.reshape(X_test_val.shape[0], num_rowstv, num_columnstv, num_channels)

    batch_size = 16
    epochs = 300
    model = Sequential()

    model.add(Conv2D(filters=emoparm[wr_emotion][0],
                     kernel_size=2,
                     input_shape=(num_rows, num_columns, num_channels),
                     activation='relu'))

    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(emoparm[wr_emotion][1]))

    model.add(Conv2D(filters=emoparm[wr_emotion][2], kernel_size=2,
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=1))
    model.add(Dropout(emoparm[wr_emotion][3]))
    model.add(GlobalAveragePooling2D())
    
    model.add(Dense(2, activation='softmax'))

    # Compile the keras model
    model.compile(loss='categorical_crossentropy',
                  optimizer=emoparm[wr_emotion][4],
                  metrics=['accuracy'])
        
    lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.9,
                                  patience=20, min_lr=0.000001)

    mcp_save = ModelCheckpoint(run_name,
                               save_best_only=True, monitor='val_accuracy',
                               mode='max')
    
    result = model.fit(X_train, y_train, batch_size=batch_size,
                       epochs=epochs, validation_data=(X_test_val, y_test_val),
                       callbacks=[mcp_save, lr_reduce], verbose=1)

    validation_acc = np.amax(result.history['val_accuracy'])

    model.load_weights(run_name)
    
    print('Best validation acc of epoch:', validation_acc)
    info_best_validation_acc = validation_acc

    predictions = model.predict(X_test, verbose=1)
    predictions_labels = []
    for predict in predictions:
        predictions_labels.append(predict.tolist().index(np.amax(predict)))
    correct = 0
    vp = 0
    fp = 0
    fn = 0
    vn = 0

    total = len(predictions_labels)
    for index in range(0, len(predictions_labels)):
        if y_labels[index] == 1:
            if predictions_labels[index] == 1:
                correct = correct + 1
                vp = vp + 1
            else:
                fn = fn + 1
        else:
            if predictions_labels[index] == 1:
                fp = fp + 1
            else:
                vn = vn + 1
                correct = correct + 1

    info_average = correct / total

    return wr_emotion, info_best_validation_acc, info_average, vp, fp, vn, fn, y_labels, y_test, predictions_labels, predictions


def translate_emotion_number(emotion_number):
        return {
            '0' : 'neu',
            '1' : 'des',
            '2' : 'med',
            '3' : 'ale',
            '4' : 'rai',
            '5' : 'sur',
            '6' : 'tri' 
        }[str(emotion_number)]

def binary_emotion_label(wb_emotion, y_all):
        for i in range(0,len(y_all)):
            if y_all[i] == wb_emotion:
                y_all[i] = 1
            else:
                y_all[i] = 0
        return y_all

data = []
new_instance = []

print ("sys.argv-str:", str(sys.argv[0:]))
tipo = sys.argv[1]
kfi = int(sys.argv[2])
kff = int(sys.argv[3])

print("Tipo dataset: TRTE-VERBO(1), TRTE-BADEM(2), TRTE-BADEM-VERBO(3), TRTE-VERBO-BADEM(4), TRTE-ALL(5):", tipo)

if tipo == "1": data_set = "TRTE-VERBO"
elif tipo == "2": data_set = "TRTE-BADEM"
elif tipo == "3": data_set = "TRTE-BADEM-VERBO"
elif tipo == "4": data_set = "TRTE-VERBO-BADEM"
elif tipo == "5": data_set = "TRTE-ALL"
else: print ("tipo deve ser de 1 a 5")

# out1 = data_set + "-" + "output1-kfold-" + str(kfold) + ".csv"
# out2 = data_set + "-" + "output2-kfold-" + str(kfold) + ".csv"
# out3 = data_set + "-" + "output3-kfold-" + str(kfold) + ".csv"
out1 = data_set + "-" + "output1.csv"
out2 = data_set + "-" + "output2.csv"
out3 = data_set + "-" + "output3.csv"

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
    y_dados0 = [int(inst_y[0]) for inst_y in y_dados]
    rg = range(kfi, kff)
    print("range(kfi, kff):", rg)
    for kfold in range(kfi, kff):
        print("kfold:", kfold)
        we = []
        ibvc = []
        ia = []
        vpi = []
        fpi = []
        vni = []
        fni = []
        yl = []
        yt = []
        pl = []
        pred = []

        outtr = "./kfold/" + data_set + "-train-idx-" + str(kfold) + ".csv"
        outte = "./kfold/" + data_set + "-test-idx-" + str(kfold) + ".csv"
        train_file = pd.read_csv(outtr, delimiter=';', header=None)
        train_x = train_file.to_numpy(dtype=int, copy=True).ravel()
        test_file = pd.read_csv(outte, delimiter=';', header=None)
        test_x = test_file.to_numpy(dtype=int, copy=True).ravel()

        X_train_d = np.take(X_dados, train_x, axis=0)
        y_train_d = np.take(y_dados0, train_x, axis=0)
        X_test_d = np.take(X_dados, test_x, axis=0)
        y_test_d = np.take(y_dados0, test_x, axis=0)

        for emo in EMOTION_NUMBERS:
            # run_name = translate_emotion_number(emo)
            run_name = data_set + "-" + str(emo) + "-kfold-" + str(kfold)
            wanted_emotion, info_best_validation_acc, info_average, vp, fp, vn, fn, y_labels, y_test, predictions_labels, predictions = \
                run(emo,run_name,X_train_d,y_train_d,X_test_d,y_test_d)
            we.append(emo)
            ibvc.append(info_best_validation_acc)
            ia.append(info_average)
            vpi.append(vp)
            fpi.append(fp)
            vni.append(vn)
            fni.append(fn)
            yl.append(y_labels.tolist())
            yt.append(y_test.tolist())
            pl.append(predictions_labels)
            pred.append(predictions.tolist())
            new_row = [kfold,we[emo],
                       ibvc[emo],
                       ia[emo],
                       vpi[emo],
                       fpi[emo],
                       vni[emo],
                       fni[emo],
                       yl[emo],
                       yt[emo],
                       pl[emo],
                       pred[emo]]
            # new_instance.append(new_row)
            # with open('output.json', 'w+') as outfile:
            #    json.dump(json.dumps(data), outfile)
            with open(out1, 'a', newline='') as file:
                writer = csv.writer(file, delimiter=';')
                writer.writerow(new_row)
    ##############################################################
        emo_prob = []
        emo_prev = []
        emo_dup = []
        for i in range(0, len(y_test)):
            emo_prob.append(0)
            emo_prev.append(99)
            emo_dup.append(0)

        for em in EMOTION_NUMBERS:
            for i in range(0, len(y_test)):
                if pred[em][i][1] > pred[em][i][0]:
                    if pred[em][i][1] > emo_prob[i]:
                        emo_prob[i] = pred[em][i][1]
                        emo_prev[i] = em
                        emo_dup[i] = 0
                    else:
                        if pred[em][i][1] == emo_prob[i]:
                            emo_dup[i] = 1
                            emo_prev[i] = 99
        fp = []
        vn = []
        fn = []
        vp = []

        for em in EMOTION_NUMBERS:
            fp.append(0)
            vn.append(0)
            fn.append(0)
            vp.append(0)

        dup = 0
        for i in range(0, len(y_test)):
            if emo_dup[i] == 1:
                dup = dup + 1
            else:
                for em in EMOTION_NUMBERS:
                    if emo_prev[i] == em:
                        # alteração feita em 11/06/2023 para testar o formato original. O yl passou a ter 0 ou 1 e não o label original.
                        # if emo_prev[i] == yl[em][i]:
                        if 1 == yl[em][i]:
                            vp[em] = vp[em] + 1
                        else:
                            fp[em] = fp[em] + 1
                    else:
                        # alteração feita em 11/06/2023 para testar o formato original. O yl passou a ter 0 ou 1 e não o label original.
                        # if yl[em][i] == em:
                        if yl[em][i] == 1:
                            fn[em] = fn[em] + 1
                        else:
                            vn[em] = vn[em] + 1
        precisao_m = 0
        sensibilidade_m = 0
        for i in EMOTION_NUMBERS:
            if vp[i] == 0:
                precisao = 0
                sensibilidade = 0
            else:
                precisao = vp[i] / (vp[i] + fp[i])
                sensibilidade = vp[i] / (vp[i] + fn[i])

            precisao_m = precisao_m + precisao
            sensibilidade_m = sensibilidade_m + sensibilidade
            new_row = [kfold, i, dup, vp[i], vn[i], fn[i], fp[i], precisao, sensibilidade, y_test_d]
            with open(out2, 'a', newline='') as file:
                writer = csv.writer(file, delimiter=';')
                writer.writerow(new_row)

        em = len(EMOTION_NUMBERS)
        if precisao_m == 0 or sensibilidade_m == 0:
           f1score_m = [data_set, kfold, "f1-scorem=", 0]
        else:
           f1score_m = [data_set, kfold, "f1-scorem=", 2*(precisao_m/em)*(sensibilidade_m/em) / (precisao_m/em + sensibilidade_m/em)]
        with open(out3, 'a', newline='') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow(f1score_m)
    ##############################################################
elif data_set in ("TRTE-BADEM-VERBO", "TRTE-VERBO-BADEM"):
    y_dadostr0 = [inst_y[0] for inst_y in y_dadostr]
    y_dadoste0 = [inst_y[0] for inst_y in y_dadoste]
    X_train_d = X_dadostr
    y_train_d = y_dadostr0
    X_test_d = X_dadoste
    y_test_d = y_dadoste0
    # X_all, y_all0 = add_padding(X_all, y_all0)
    # print('y_all0:', y_all0)
    #for train_idx, test_idx in strat_kfold.split(X_dados, y_dados0):
    # for kfold in range(0, 9):

    we = []
    ibvc = []
    ia = []
    vpi = []
    fpi = []
    vni = []
    fni = []
    yl = []
    yt = []
    pl = []
    pred = []

    for emo in EMOTION_NUMBERS:
        run_name = translate_emotion_number(emo)
        #run_name = data_set + "-" + str(emo)
        wanted_emotion, info_best_validation_acc, info_average, vp, fp, vn, fn, y_labels, y_test, predictions_labels, predictions = \
            run(emo, run_name, X_train_d, y_train_d, X_test_d, y_test_d)
        we.append(emo)
        ibvc.append(info_best_validation_acc)
        ia.append(info_average)
        vpi.append(vp)
        fpi.append(fp)
        vni.append(vn)
        fni.append(fn)
        yl.append(y_labels.tolist())
        yt.append(y_test.tolist())
        pl.append(predictions_labels)
        pred.append(predictions.tolist())
        new_row = [0, we[emo],
                   ibvc[emo],
                   ia[emo],
                   vpi[emo],
                   fpi[emo],
                   vni[emo],
                   fni[emo],
                   yl[emo],
                   yt[emo],
                   pl[emo],
                   pred[emo]]
        # new_instance.append(new_row)
        # with open('output.json', 'w+') as outfile:
        #    json.dump(json.dumps(data), outfile)
        with open(out1, 'a', newline='') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow(new_row)
    ###############################################################################
    emo_prob = []
    emo_prev = []
    emo_dup = []
    for i in range(0, len(y_test)):
        emo_prob.append(0)
        emo_prev.append(99)
        emo_dup.append(0)

    for em in EMOTION_NUMBERS:
        for i in range(0, len(y_test)):
            if pred[em][i][1] > pred[em][i][0]:
                if pred[em][i][1] > emo_prob[i]:
                    emo_prob[i] = pred[em][i][1]
                    emo_prev[i] = em
                    emo_dup[i] = 0
                else:
                    if pred[em][i][1] == emo_prob[i]:
                        emo_dup[i] = 1
                        emo_prev[i] = 99
    fp = []
    vn = []
    fn = []
    vp = []

    for em in EMOTION_NUMBERS:
        fp.append(0)
        vn.append(0)
        fn.append(0)
        vp.append(0)

    dup = 0
    for i in range(0, len(y_test)):
        if emo_dup[i] == 1:
            dup = dup + 1
        else:
            for em in EMOTION_NUMBERS:
                if emo_prev[i] == em:
                    # alteração feita em 13/08/2023 para corrigir como foi feito em 11/06/2023 para os demais datasets
                    # BADEM, VERBO e ALL
                    # if emo_prev[i] == yl[em][i]:
                    if 1 == yl[em][i]:
                        vp[em] = vp[em] + 1
                    else:
                        fp[em] = fp[em] + 1
                else:
                    # alteração feita em 13/08/2023 para corrigir como foi feito em 11/06/2023 para os demais datasets
                    # BADEM, VERBO e ALL
                    # if yl[em][i] == em:
                    if yl[em][i] == 1:
                        fn[em] = fn[em] + 1
                    else:
                        vn[em] = vn[em] + 1
    precisao_m = 0
    sensibilidade_m = 0
    for i in EMOTION_NUMBERS:
        if vp[i] == 0:
            precisao = 0
            sensibilidade = 0
        else:
            precisao = vp[i] / (vp[i] + fp[i])
            sensibilidade = vp[i] / (vp[i] + fn[i])

        precisao_m = precisao_m + precisao
        sensibilidade_m = sensibilidade_m + sensibilidade
        new_row = [0, i, dup, vp[i], vn[i], fn[i], fp[i], precisao, sensibilidade, y_test_d]
        with open(out2, 'a', newline='') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow(new_row)

    em = len(EMOTION_NUMBERS)

    if precisao_m == 0 or sensibilidade_m == 0:
        f1score_m = [data_set, 0, "f1-scorem=", 0]
    else:
        f1score_m = [data_set, 0, "f1-scorem=",
                     2 * (precisao_m / em) * (sensibilidade_m / em) / (precisao_m / em + sensibilidade_m / em)]

    with open(out3, 'a', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow(f1score_m)
