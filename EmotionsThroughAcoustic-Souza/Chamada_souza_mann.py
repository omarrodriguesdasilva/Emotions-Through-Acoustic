# Chamada_souza_mann.py: test mann whitney for souza features.

import os
import keras
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wave
from tqdm import tqdm
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.models import Sequential, model_from_json
from keras.layers import Dense, GRU
from keras.layers import Flatten, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D
from keras.utils import np_utils
from keras import losses
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
# incluido por Omar devido ao erro de execução - 09/10/2023
# AttributeError: module 'keras.optimizers' has no attribute 'RMSprop'

from tensorflow import keras
from keras import optimizers

#  incluido por Omar - adaptação às rotinas de teste e execução teste estatístico mannwhitney
import sys
import csv
from scipy.stats import mannwhitneyu
aug_type = "none"
config = {
    "sample_rate": 44100,
    "audio_duration": 3,
    "n_mels": 128,
    "n_fft": 1024,
    "n_mfcc": 20,
    "fmin": None,
    "fmax": 8000,
    "dct_type": 2,
    # numero de classes ajustado de 4 (original) para 7 (datasets verbo e badem)
    # "n_classes": 4,
    "n_classes": 7,
    "learning_rate": 0.0001,
    "max_epochs": 100,
    "batch_size": 64,
    "n_splits": 1,
    "datadir": "Audio_Speech_Actors_01−24\\",
    "actors_tests": [1, 2],
    "dnn_type": "rnn-gru"  # cnn | rnn-gru
}


EMOTION_NUMBERS = [0, 1, 2, 3, 4, 5, 6]
# Parâmetros de treinamento para cada emoção:
#            '0' : 'neu'
#            '1' : 'des'
#            '2' : 'med'
#            '3' : 'ale'
#            '4' : 'rai'
#            '5' : 'sur'
#            '6' : 'tri' 

def __init__(self, dataset_path=None):
    self.dataset_path = dataset_path


def noise(data):
    noise_amp = 0.025 * np.random.uniform() * np.amax(data)
    data = data.astype("float64") + noise_amp * np.random. normal(size=data.shape[0])
    return data


def shift(data):
    s_range = int(np.random.uniform(low=-5, high=5) * 500)
    return np.roll(data, s_range)


def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(data, rate)


def pitch(data, sample_rate):
    pitch_pm = 2
    pitch_change = pitch_pm * 2 * (np.random.uniform())
    data = librosa.effects.pitch_shift(data.astype("float64"), sr=sample_rate, n_steps=pitch_change)
    return data


def speed_n_pitch(data):
    length_change = np.random.uniform(low=0.8, high=1)
    speed_fac = 1.0 / length_change
    tmp = np.interp(np.arange(0, len(data), speed_fac), np.arange(0, len(data)), data)
    minlen = min(data.shape[0], tmp.shape[0])
    data *= 0
    data[0:minlen] = tmp[0:minlen]
    return data


def aug_neutral_with_stretch(audio, path_new_file):
    import soundfile as sf
    if not os.path.exists(path_new_file):
        audio, sr = sf.read(audio)
        audio = noise(audio)
        sf.write(path_new_file, audio, sr, "PCM_16")


def plot_time_series(data):
    fig = plt.figure(figsize=(14, 8))
    plt.title("Raw_wave_")
    plt.ylabel("Amplitude")
    plt.plot(np.linspace(0, 1, len(data)), data)
    plt.show()


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


def create_data_frame():
    file_paths = get_all_filepaths(config["datadir"])
    data_df = pd.DataFrame(
        columns=["path", "source", "actor", "gender", "intensity", "statement", "repetition", "emotion_id",
                 "emotion_name", "n_frames", "label"], dtype="float")

    #  dir_list = os.listdir(config["datadir"])
    dir_list = file_paths
    #  dir_list.sort()
    count = 0
    for i in dir_list:
        #  file_list = os.listdir(config["datadir"] + i)
        if config["datadir"] == "audio/verbo":
            emotion = translate_emotion_verbo(i)
        elif config["datadir"] == "audio/badem":
            emotion = translate_emotion_badem(i)
        else:
            print("ERRO-INCORRETO DATASET ESPECIFICADO")
        # filename = i.split('/')[-1]
        # token = i.split('-')[0]
        emotion = int(emotion)
        label = str(emotion)
        emotion_name = label
        lb = label
        path = i
        n_frames = wave.open(i).getnframes()
        gender = "gender"
        intensity = 99
        statement = 88
        repeat = 77
        src = "src"
        actor = "actor"
        data_df.loc[count] = [path, src, actor, gender, intensity,
                              statement, repeat, emotion, emotion_name, n_frames, lb]
        count += 1
    return data_df


def load_dataset():
    data_df = create_data_frame()
    data2_df = data_df.copy()
    data2_df = data2_df[data2_df.label != "none"].reset_index(drop=True)
    data = pd.DataFrame(columns=["feature"])
    for i in tqdm(range(len(data2_df))):
        audio_time_series, sr = librosa.load(data2_df.path[i], sr=config["sample_rate"], offset=0.5,
                                             duration=config["audio_duration"], res_type="kaiser_fast")
        mfcc = librosa.feature.mfcc(y=audio_time_series, sr=config["sample_rate"], n_mfcc=config["n_mfcc"],
                                    n_fft=config["n_fft"], n_mels=config["n_mels"], fmax=config["fmax"])
        mfcc_feature = np.mean(mfcc.T, axis=0)
        data.loc[i] = [mfcc_feature]

    df3 = pd.DataFrame(data["feature"].values.tolist())
    labels = data2_df.label
    newdf = pd.concat([df3, labels], axis=1)
    rnewdf = newdf.rename(index=str, columns={"0": "label"})
    rnewdf = rnewdf.fillna(0)

    for i in range(len(rnewdf)):
        X_dados1 = np.concatenate((config["datadir"], rnewdf.iloc[i]), axis=None)
        new_row = X_dados1
        with open(out1, 'a', newline='') as file:
             writer = csv.writer(file, delimiter=';')
             writer.writerow(new_row)
    aug_type = "none"
    # aug_df = augmented_data(aug_type=[])
    aug_df = []
    if aug_df:
        aug_df.append(rnewdf)
        combined_df = pd.concat(aug_df, ignore_index=True, sort=False)
    else:
        combined_df = pd.concat([rnewdf], ignore_index=True, sort=False)
    combined_df = combined_df.fillna(0)
    X = combined_df.drop(["label"], axis=1)
    y = combined_df.label
    if config["dnn_type"] == "rnn-gru":
        X = X.values.reshape((X.shape[0], 1, X.shape[1]))
        print("X.shape:", X.shape)
    return X, y


def build_rnn_1gru(x_traincnn):
    print("input_shape:", x_traincnn.shape)
    input_shape = (x_traincnn.shape[1], x_traincnn.shape[2])
    dropout = 0.2
    rd = 0.1
    n_dim = 128
    model = Sequential()
    model.add(GRU(n_dim, input_shape=input_shape, dropout=dropout,
                  recurrent_dropout=rd, return_sequences=False))
    model.add(Dense(config["n_classes"]))
    model.add(Activation("softmax"))
    opt = keras.optimizers.RMSprop(learning_rate=0.0001, decay=0.0)
    model.summary()
    model.compile(loss=losses.categorical_crossentropy, optimizer=opt,
                  metrics=["acc"])
    return model, opt



def print_confusion_matrix(confusion_matrix, class_names, acc_score,
                           graph_name, aug_type, figsize=(10, 7), fontesize=14):
    confusion_matrix = confusion_matrix.astype("float") / confusion_matrix.sum(axis=1)[:, np.newaxis]
    # df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names,    )
    df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names)
    fig = plt.figure(figsize=figsize)
    heatmap = sns.heatmap(df_cm, annot=True, fmt=".2%")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha="right", fontsize=fontesize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha="right", fontsize=fontesize)
    plt.ylabel("True_label")
    plt.xlabel("Predicted_label")
    plt.title(f"{graph_name}_Accuracy_{acc_score:.2f}%")
    plt.savefig(f"tests/accuracy/{graph_name}_{acc_score:.2f}_{aug_type}.png")
    plt.show()


def predict_test(dataset, loaded_model, y_test, lb, dataset_type, aug_type):
    preds = loaded_model.predict(dataset, batch_size=config["batch_size"], verbose=1)
    preds1 = preds.argmax(axis=1)
    abc = preds1.astype(int).flatten()
    predictions = (lb.inverse_transform(abc))
    preddf = pd.DataFrame({"predictedvalues": predictions})
    actual = y_test.argmax(axis=1)
    abc123 = actual.astype(int).flatten()
    actualvalues = (lb.inverse_transform(abc123))
    actualdf = pd.DataFrame({"actualvalues": actualvalues})
    finaldf = actualdf.join(preddf)
    finaldf.to_csv(f"tests/accuracy/predictions_{dataset_type}.csv", index=False)
    classes = finaldf.actualvalues.unique()
    classes.sort()
    y_true = finaldf.actualvalues
    y_pred = finaldf.predictedvalues
    acc_score = accuracy_score(y_true, y_pred) * 100
    print(f"\nacc_score:_{acc_score:.2f}%")
    clf_report = classification_report(y_true, y_pred, output_dict=True)
    print(clf_report)
    sns.heatmap(pd.DataFrame(clf_report).iloc[: -1, :].T, annot=True, fmt=".2%")
    plt.title(f"report_{dataset_type}")
    plt.savefig(f"tests/accuracy/report_{dataset_type}_{acc_score:.2f}_{aug_type}.png")
    return y_true, y_pred


data = []
new_instance = []
data_set = "TRTE-BADEM-VERBO"
out1 = data_set + "-" + "output1.csv"
config["datadir"] = "audio/badem"
X_dados_badem, y_dadoste = load_dataset()

config["datadir"] = "audio/verbo"
X_dados_verbo, y_dadostr = load_dataset()

## o vetor X_dados_badem contém na posição [i][0] o vetor de dados.
## A posição [0] é devido ao dataframe que tiveram as demais posições removidas.
print("len(X_dados_badem[0][0]:", len(X_dados_badem[0][0]))
for i in range(len(X_dados_badem[0][0])):
    grupo_badem = []
    for j in range(len(X_dados_badem)):
        grupo_badem.append(X_dados_badem[j][0][i])
    print("grupo_badem1:", grupo_badem)
    grupo_badem.sort()
    # print("grupo_badem2:", grupo_badem)
    grupo_verbo = []
    for j in range(len(X_dados_verbo)):
        grupo_verbo.append(X_dados_verbo[j][0][i])
    # print("grupo_verbo1:", grupo_verbo)
    grupo_badem.sort()
    # print("grupo_verbo2:", grupo_verbo)
    estatistica_U, p_valor = mannwhitneyu(grupo_badem, grupo_verbo)
    print(f"Variável {i}: Estatística U = {estatistica_U}, Valor p = {p_valor}")

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