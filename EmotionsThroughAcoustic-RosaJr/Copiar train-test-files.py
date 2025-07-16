## copiar train-test-files.py: train and test RosaJr algorithm for each kfold

import sys
import os
import csv
import math
import random

import numpy as np
import pandas as pd

import main_SVM as msvm

EMOTION_NUMBERS = [0, 1, 2, 3, 4, 5, 6]


def translate_emotion_badem(file_path):
    filename = file_path.split('/')[-1]
    token = filename.split('_')[2]
    return {
        'N': 0,
        'J': 1,
        'M': 2,
        'F': 3,
        'R': 4,
        'S': 5,
        'T': 6
    }[token]


def translate_emotion_verbo(file_path):
    filename = file_path.split('/')[-1]
    token = filename.split('-')[0]

    return {
        'neu': 0,
        'des': 1,
        'med': 2,
        'ale': 3,
        'rai': 4,
        'sur': 5,
        'tri': 6
    }[token]


def generate_copy_command(origem, destino):
    filename = origem.split('/')[-1]
    return "copy {0} {1}".format(origem, destino + filename)
    # return "cp {0} {1}".format(origem, destino + filename)


def get_all_filepaths(path):
    result_filepaths = []
    for inst in os.listdir(path):
        recursive_file_instances = []
        if os.path.isdir("{}/{}".format(path, inst)):
            recursive_file_instances = get_all_filepaths("{}/{}".format(path, inst))
            for filepath in recursive_file_instances:
                result_filepaths.append(filepath)
        else:
            result_filepaths.append("{}/{}".format(path, inst))

    return result_filepaths + recursive_file_instances


def copy_files(path):
    filepaths = get_all_filepaths(path)
    i = 0
    for filepath in filepaths:
        print("filepath:", filepath)
        filetype = filepath.split('/')[-2]
        print('filetype:', filetype)
        if filetype == 'badem':
            emodir = translate_emotion_badem(filepath)
        else:
            emodir = translate_emotion_verbo(filepath)
        if i in test_x:
            output_path = output_path_test[emodir] + '/'
        else:
            output_path = output_path_train[emodir] + '/'
        copy_command = generate_copy_command(filepath, output_path)
        copy_command = copy_command.replace(barant, barnew)
        # print("barant:", barant)
        # print("barnew:", barnew)
        print(copy_command)
        os.system(copy_command)
        i = i + 1


def copy_files_train_all(path):
    filepaths = get_all_filepaths(path)
    i = 0
    for filepath in filepaths:
        print("filepath:", filepath)
        filetype = filepath.split('/')[-2]
        print('filetype:', filetype)
        if filetype == 'badem':
            emodir = translate_emotion_badem(filepath)
        else:
            emodir = translate_emotion_verbo(filepath)
        output_path = output_path_train[emodir] + '/'
        copy_command = generate_copy_command(filepath, output_path)
        copy_command = copy_command.replace(barant, barnew)
        print("barant:", barant)
        print("barnew:", barnew)
        print(copy_command)
        os.system(copy_command)
        i = i + 1


def copy_dataset(path, dataset):
    filepaths = get_all_filepaths(path)
    for filepath in filepaths:
        print("filepath:", filepath)
        filetype = filepath.split('/')[-2]
        print('filetype:', filetype)
        if filetype == 'badem':
            emodir = translate_emotion_badem(filepath)
            if dataset == "TRTE-BADEM-VERBO":
                output_path = output_path_train[emodir] + '/'
            else:
                output_path = output_path_test[emodir] + '/'
        else:
            emodir = translate_emotion_verbo(filepath)
            if dataset == "TRTE-BADEM-VERBO":
                output_path = output_path_test[emodir] + '/'
            else:
                output_path = output_path_train[emodir] + '/'
        copy_command = generate_copy_command(filepath, output_path)
        copy_command = copy_command.replace(barant, barnew)
        # print("barant:", barant)
        # print("barnew:", barnew)
        print(copy_command)
        os.system(copy_command)


print("sys.argv-str:", str(sys.argv[0:]))
tipo = sys.argv[1]
kfi = int(sys.argv[2])
kff = int(sys.argv[3])
print("Tipo dataset: TRTE-VERBO(1), TRTE-BADEM(2), TRTE-BADEM-VERBO(3), TRTE-VERBO-BADEM(4), TRTE-ALL(5), TRTE-ALL(6):", tipo)
print("os.sep:", os.sep)
if tipo == "1": data_set = "TRTE-VERBO"
elif tipo == "2": data_set = "TRTE-BADEM"
elif tipo == "3": data_set = "TRTE-BADEM-VERBO"
elif tipo == "4": data_set = "TRTE-VERBO-BADEM"
elif tipo == "5": data_set = "TRTE-ALL"
elif tipo == "6": data_set = "TRTE-TRAIN"
else: print("tipo deve ser de 1 a 6")

output_path_test = []
barant = '/'
barnew = '\\'
rmdel = 'del '
audiodir = 'audiocorr'
output_path_test.append("{}{}{}".format("test", os.sep, "Neutro"))
output_path_test.append("{}{}{}".format("test", os.sep, "Desgosto"))
output_path_test.append("{}{}{}".format("test", os.sep, "Medo"))
output_path_test.append("{}{}{}".format("test", os.sep, "Alegria"))
output_path_test.append("{}{}{}".format("test", os.sep, "Raiva"))
output_path_test.append("{}{}{}".format("test", os.sep, "Surpresa"))
output_path_test.append("{}{}{}".format("test", os.sep, "Tristeza"))
output_path_train = []
output_path_train.append("{}{}{}".format("train", os.sep, "Neutro"))
output_path_train.append("{}{}{}".format("train", os.sep, "Desgosto"))
output_path_train.append("{}{}{}".format("train", os.sep, "Medo"))
output_path_train.append("{}{}{}".format("train", os.sep, "Alegria"))
output_path_train.append("{}{}{}".format("train", os.sep, "Raiva"))
output_path_train.append("{}{}{}".format("train", os.sep, "Surpresa"))
output_path_train.append("{}{}{}".format("train", os.sep, "Tristeza"))

if data_set in ("TRTE-BADEM", "TRTE-VERBO", "TRTE-ALL"):
    for kfold in range(kfi, kff):
        for i in output_path_train:
            os.system(rmdel + i + barnew + "*.*")
            os.system("rmdir " + i)
            os.system("mkdir " + i)
        for i in output_path_test:
            os.system(rmdel + i + barnew + "*.*")
            os.system("rmdir " + i)
            os.system("mkdir " + i)
        outtr = "./kfold/" + data_set + "-train-idx-" + str(kfold) + ".csv"
        outte = "./kfold/" + data_set + "-test-idx-" + str(kfold) + ".csv"
        train_file = pd.read_csv(outtr, delimiter=';', header=None)
        train_x = train_file.to_numpy(dtype=int, copy=True).ravel()
        test_file = pd.read_csv(outte, delimiter=';', header=None)
        test_x = test_file.to_numpy(dtype=int, copy=True).ravel()

        if data_set == "TRTE-BADEM":
            AUDIO_PATH = audiodir + '/badem'
        elif data_set == "TRTE-VERBO":
            AUDIO_PATH = audiodir + '/verbo'
        elif data_set == "TRTE-ALL":
            AUDIO_PATH = audiodir
        copy_files(AUDIO_PATH)
        msvm.main_SVM(data_set, kfold)
elif data_set == "TRTE-TRAIN":
    AUDIO_PATH = audiodir
    copy_files_train_all(AUDIO_PATH)
    msvm.main_SVM(data_set, 0)
else:
    for i in output_path_train:
        print(rmdel + i + barnew + "*.*")
        os.system(rmdel + i + barnew + "*.*")
        os.system("rmdir " + i)
        os.system("mkdir " + i)
    for i in output_path_test:
        print(rmdel + i + barnew + "*.*")
        os.system(rmdel + i + barnew + "*.*")
        os.system("rmdir " + i)
        os.system("mkdir " + i)
    # Alteração em 12/01/2023: foi efetuada padronização nos arquivos de audio pois alguns estavam diferentes.
    # Foram alterados: acodec=pcm_s16le (de pcm+s32le), mono (de stereo), ar=44100 Hz (de 16100 Hz).
    # AUDIO_PATH = 'audio'
    AUDIO_PATH = 'audiocorr'
    copy_dataset(AUDIO_PATH, data_set)
    msvm.main_SVM(data_set, 0)