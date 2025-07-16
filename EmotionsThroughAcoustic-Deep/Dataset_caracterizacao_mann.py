# test mann-whitney

import sys
import csv
import math
import random

import numpy as np

from dataset_loader import DatasetLoader
from dataset_loaderv import DatasetLoaderv

import pandas as pd
from scipy.stats import mannwhitneyu


data = []
new_instance = []

X_dados_verbo, y_dadostr = load_datasetv()
X_dados_badem, y_dadoste = load_dataset()

X_dados1 = []
for i in range(len(X_dados_badem[0][0])):
    grupo_badem = []
    for j in range(len(X_dados_badem)):
        for k in range(len(X_dados_badem[j])):
            grupo_badem.append(X_dados_badem[j][k][i])
    grupo_badem.sort()
    grupo_verbo = []
    for j in range(len(X_dados_verbo)):
        for k in range(len(X_dados_verbo[j])):
            grupo_verbo.append(X_dados_verbo[j][k][i])
    grupo_badem.sort()
    estatistica_U, p_valor = mannwhitneyu(grupo_badem, grupo_verbo)
    # Testa a hipótese nula comumente usando um nível de significância de 0,05
    nivel_significancia = 0.05
    if p_valor < nivel_significancia:
        print(f"Rejeita a hipótese nula para a variável {i}: Há evidências de diferença entre os grupos.")
    else:
        print(
            f"Não rejeita a hipótese nula para a variável {i}: Não há evidências suficientes de diferença entre os grupos.")
    print("\n")
    outy = "data_sety" + "_" + "carac.csv"
    new_row = np.concatenate((estatistica_U, p_valor), axis=None)
    with open(outy, 'a', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow(new_row)