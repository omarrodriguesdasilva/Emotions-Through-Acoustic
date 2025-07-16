## geracao caracteristicas.py: test mann whitney for RosaJr features. 

import sys
import os
import csv
import math
import random

import numpy as np
import pandas as pd

import main_SVM_carac as msvm
from scipy.stats import mannwhitneyu

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
    filepathso = origem.replace(exec_linux, exec_windows)
    return "copy {0} {1}".format(filepathso, destino + filename)
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
        # print("filepath:", filepath)
        filetype = filepath.split('/')[-2]
        # print('filetype:', filetype)
        if filetype == 'badem':
            emodir = translate_emotion_badem(filepath)
        else:
            emodir = translate_emotion_verbo(filepath)

        output_path = output_path_train[emodir] + '/'
        copy_command = generate_copy_command(filepath, output_path)
        # print(copy_command)
        os.system(copy_command)
        i = i + 1


def copy_dataset(path, dataset):
    filepaths = get_all_filepaths(path)
    for filepath in filepaths:
        # print("filepath:", filepath)
        filetype = filepath.split('/')[-2]
        # print('filetype:', filetype)
        if filetype == 'badem':
            emodir = translate_emotion_badem(filepath)
            if dataset == "TRTE-BADEM-VERBO":
                output_path = output_path_train[emodir] + exec_bar
            else:
                output_path = output_path_test[emodir] + exec_bar
        else:
            emodir = translate_emotion_verbo(filepath)
            if dataset == "TRTE-BADEM-VERBO":
                output_path = output_path_test[emodir] + exec_bar
            else:
                output_path = output_path_train[emodir] + exec_bar
        copy_command = generate_copy_command(filepath, output_path)
        # print(copy_command)
        os.system(copy_command)


print("sys.argv-str:", str(sys.argv[0:]))
tipo = sys.argv[1]
kfi = int(sys.argv[2])
kff = int(sys.argv[3])
print("Tipo dataset: TRTE-VERBO(1), TRTE-BADEM(2), TRTE-BADEM-VERBO(3), TRTE-VERBO-BADEM(4), TRTE-ALL(5):", tipo)

if tipo == "1": data_set = "TRTE-VERBO"
elif tipo == "2": data_set = "TRTE-BADEM"
elif tipo == "3": data_set = "TRTE-BADEM-VERBO"
elif tipo == "4": data_set = "TRTE-VERBO-BADEM"
elif tipo == "5": data_set = "TRTE-ALL"
else: print("tipo deve ser de 1 a 5")

exec_windows = "\\"
exec_linux = "/"
exec_bar = exec_windows
output_path_test = []
output_path_test.append("{}{}{}".format("test", exec_bar, "Neutro"))
output_path_test.append("{}{}{}".format("test", exec_bar, "Desgosto"))
output_path_test.append("{}{}{}".format("test", exec_bar, "Medo"))
output_path_test.append("{}{}{}".format("test", exec_bar, "Alegria"))
output_path_test.append("{}{}{}".format("test", exec_bar, "Raiva"))
output_path_test.append("{}{}{}".format("test", exec_bar, "Surpresa"))
output_path_test.append("{}{}{}".format("test", exec_bar, "Tristeza"))
output_path_train = []
output_path_train.append("{}{}{}".format("train", exec_bar, "Neutro"))
output_path_train.append("{}{}{}".format("train", exec_bar, "Desgosto"))
output_path_train.append("{}{}{}".format("train", exec_bar, "Medo"))
output_path_train.append("{}{}{}".format("train", exec_bar, "Alegria"))
output_path_train.append("{}{}{}".format("train", exec_bar, "Raiva"))
output_path_train.append("{}{}{}".format("train", exec_bar, "Surpresa"))
output_path_train.append("{}{}{}".format("train", exec_bar, "Tristeza"))

for i in output_path_train:
    ## os.system("rm " + i + "/*.*")
    print("del " + i + "\\*.*")
    os.system("del " + i + "\\*.*")
    print("rmdir " + i)
    os.system("rmdir " + i)
    print("mkdir " + i)
    os.system("mkdir " + i)

AUDIO_PATH = 'audio/badem'
data_set = "TRTE-BADEM-VERBO"
copy_dataset(AUDIO_PATH, data_set)
carac_vetor_badem = msvm.main_SVM_carac(data_set, 0)
print("carac_vetor_badem1:", carac_vetor_badem)
carac_vetor_badem = np.array(carac_vetor_badem)
print("carac_vetor_badem2:", carac_vetor_badem)

for i in output_path_train:
    ## os.system("rm " + i + "/*.*")
    print("del " + i + "\\*.*")
    os.system("del " + i + "\\*.*")
    print("rmdir " + i)
    os.system("rmdir " + i)
    print("mkdir " + i)
    os.system("mkdir " + i)
AUDIO_PATH = 'audio/verbo'
data_set = "TRTE-VERBO-BADEM"
copy_dataset(AUDIO_PATH, data_set)
carac_vetor_verbo = np.array(msvm.main_SVM_carac(data_set, 0))

for i in range(2, len(carac_vetor_badem[0])):
    grupo_badem = []
    grupo_verbo = []
    for j in range(len(carac_vetor_badem)):
        grupo_badem.append(float(carac_vetor_badem[j][i]))
    print("grupo_badem1:", grupo_badem)
    grupo_badem.sort()
    print("grupo_badem2:", grupo_badem)
    for j in range(len(carac_vetor_verbo)):
        grupo_verbo.append(float(carac_vetor_verbo[j][i]))
    print("grupo_verbo1:", grupo_verbo)
    grupo_verbo.sort()
    print("grupo_verbo2:", grupo_verbo)
    estatistica_U, p_valor = mannwhitneyu(grupo_badem, grupo_verbo)

    # Testa a hipótese nula comumente usando um nível de significância de 0,05
    nivel_significancia = 0.05
    if p_valor < nivel_significancia:
       print(f"Rejeita a hipótese nula para a variável {i}: Há evidências de diferença entre os grupos.")
    else:
       print(f"Não rejeita a hipótese nula para a variável {i}: Não há evidências suficientes de diferença entre os grupos.")
       print("\n")
    outy = "data_sety" + "_" + "carac.csv"
    new_row = np.concatenate((estatistica_U, p_valor), axis=None)
    with open(outy, 'a', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow(new_row)