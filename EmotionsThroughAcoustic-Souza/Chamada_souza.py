# Chamada_souza.py: train and test souza algorithm
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

import sys
import csv

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


def aug_noise(syn_data1):
    for i in tqdm(range(len(data2_df))):
        audio_time_series, sr = librosa.load(data2_df.path[i], sr=config["sample_rate"], res_type="kaiser_fast",
                                             duration=config["audio_duration"], offset=0.5)
        audio_time_series = noise(audio_time_series)
        mfcc = librosa.feature.mfcc(y=audio_time_series, sr=config["sample_rate"], n_mfcc=config["n_mfcc"],
                                    n_fft=config["n_fft"],  # passado para mel spectrogram
                                    n_mels=config["n_mels"], fmax=config["fmax"])  # passado para mel
        mfcc_feature = np.mean(mfcc.T, axis=0)
        syn_data1.loc[i] = [mfcc_feature, data2_df.label[i]]


def aug_pitch(syn_data2):
    for i in tqdm(range(len(data2_df))):
        audio_time_series, sr = librosa.load(data2_df.path[i], sr=config["sample_rate"], res_type="kaiser_fast",
                                             duration=config["audio_duration"], offset=0.5)
        audio_time_series = pitch(audio_time_series, sr)
        mfcc = librosa.feature.mfcc(y=audio_time_series, sr=config["sample_rate"], n_mfcc=config["n_mfcc"],
                                    n_fft=config["n_fft"],  # passado para mel spectrogram
                                    n_mels=config["n_mels"], fmax=config["fmax"])  # passado para mel
        mfcc_feature = np.mean(mfcc.T, axis=0)
        syn_data2.loc[i] = [mfcc_feature, data2_df.label[i]]


def aug_speed_pitch(syn_data3):
    for i in tqdm(range(len(data2_df))):
        audio_time_series, sr = librosa.load(data2_df.path[i], sr=config["sample_rate"], res_type="kaiser_fast",
                                             duration=config["audio_duration"], offset =0.5)
        audio_time_series = speed_n_pitch(audio_time_series)
        mfcc = librosa.feature.mfcc(y=audio_time_series, sr=config["sample_rate"], n_mfcc=config["n_mfcc"],
                                    n_fft=config["n_fft"],  # passado para mel spectrogram
                                    n_mels=config["n_mels"], fmax=config["fmax"])  # passado para mel
        mfcc_feature = np.mean(mfcc.T, axis=0)
        syn_data3.loc[i] = [mfcc_feature, data2_df.label[i]]


def aug_stretch(syn_data5):
    for i in tqdm(range(len(data2_df))):
        audio_time_series, sr = librosa.load(data2_df.path[i], sr=config["sample_rate"], res_type="kaiser_fast",
                                             duration=config["audio_duration"], offset=0.5)
        audio_time_series = stretch(audio_time_series)
        mfcc = librosa.feature.mfcc(y=audio_time_series, sr=config["sample_rate"], n_mfcc=config["n_mfcc"],
                                    n_fft=config["n_fft"],  # passado para mel spectrogram
                                    n_mels=config["n_mels"], fmax=config["fmax"])  # passado para mel
        mfcc_feature = np.mean(mfcc.T, axis=0)
        syn_data5.loc[i] = [mfcc_feature, data2_df.label[i]]


def augmented_data(aug_type):
    aug_df = []
    if "noise" in aug_type:
        syn_data1 = pd.DataFrame(columns=["feature", "label"])
        aug_noise(syn_data1)
        syn_data1 = syn_data1.reset_index(drop=True)
        df4 = pd.DataFrame(syn_data1["feature"].values.tolist())
        labels4 = syn_data1.label
        syndf1 = pd.concat([df4, labels4], axis=1)
        syndf1 = syndf1.rename(index=str, columns={"0": "label"})
        syndf1 = syndf1.fillna(0)
        aug_df.append(syndf1)
    if "pitch" in aug_type:
        syn_data2 = pd.DataFrame(columns=["feature", "label"])
        aug_pitch(syn_data2)
        syn_data2 = syn_data2.reset_index(drop=True)
        df4 = pd.DataFrame(syn_data2["feature"].values.tolist())
        labels4 = syn_data2.label
        syndf2 = pd.concat([df4, labels4], axis=1)
        syndf2 = syndf2.rename(index=str, columns={"0": "label"})
        syndf2 = syndf2.fillna(0)
        aug_df.append(syndf2)
    if "speed_pitch" in aug_type:
        syn_data3 = pd.DataFrame(columns=["feature", "label"])
        aug_speed_pitch(syn_data3)
        syn_data3 = syn_data3.reset_index(drop=True)
        df4 = pd.DataFrame(syn_data3["feature"].values.tolist())
        labels4 = syn_data3.label
        syndf3 = pd.concat([df4, labels4], axis=1)
        syndf3 = syndf3.rename(index=str, columns={"0": "label"})
        syndf3 = syndf3.fillna(0)
        aug_df.append(syndf3)
    if "stretch" in aug_type:
        syn_data5 = pd.DataFrame(columns=["feature", "label"])
        aug_stretch(syn_data5)
        syn_data5 = syn_data5.reset_index(drop=True)
        df4 = pd.DataFrame(syn_data5["feature"].values.tolist())
        labels4 = syn_data5.label
        syndf5 = pd.concat([df4, labels4], axis=1)
        syndf5 = syndf5.rename(index=str, columns={"0": "label"})
        syndf5 = syndf5.fillna(0)
        aug_df.append(syndf5)
    return aug_df


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
    aug_df = augmented_data(aug_type=[])
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

print ("sys.argv-str:", str(sys.argv[0:]))
tipo = sys.argv[1]
kfi = int(sys.argv[2])
kff = int(sys.argv[3])
#emo = int(sys.argv[3])
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
    config["datadir"] = "audio/badem"    
    X_dados, y_dados = load_dataset()
elif data_set == "TRTE-VERBO":
    config["datadir"] = "audio/verbo"
    X_dados, y_dados = load_dataset()
elif data_set == "TRTE-VERBO-BADEM":
    config["datadir"] = "audio/verbo"
    X_dadostr, y_dadostr = load_dataset()
    config["datadir"] = "audio/badem"       
    X_dadoste, y_dadoste = load_dataset()
elif data_set == "TRTE-BADEM-VERBO":
    config["datadir"] = "audio/verbo"    
    X_dadoste, y_dadoste = load_dataset()
    config["datadir"] = "audio/badem" 
    X_dadostr, y_dadostr = load_dataset()
elif data_set == "TRTE-ALL":
    config["datadir"] = "audio/badem"
    X_dados, y_dados = load_dataset()
    config["datadir"] = "audio/verbo"
    X_dadosv, y_dadosv = load_dataset()

    X_dados = np.append(X_dados, X_dadosv, axis=0)
    # X_dados = pd.concat(X_dados, X_dadosv)
    # y_dados = pd.concat(y_dados, y_dadosv)
    y_dados = np.append(y_dados, y_dadosv, axis=0)
    # X_dados = np.asarray(X_dados)
    # y_dados = np.asarray(y_dados)

else:
    print ("Invalido dataset especificado.")

if data_set in ("TRTE-BADEM", "TRTE-VERBO", "TRTE-ALL"):
    print("X_dados.shape:", X_dados.shape)
    rg = range(kfi, kff)
    print("range(kfi, kff):", rg)
    for kfold in range(kfi, kff):
        print("kfold:", kfold)
        outtr = "./kfold/" + data_set + "-train-idx-" + str(kfold) + ".csv"
        outte = "./kfold/" + data_set + "-test-idx-" + str(kfold) + ".csv"
        train_file = pd.read_csv(outtr, delimiter=';', header=None)
        train_x = train_file.to_numpy(dtype=int, copy=True).ravel()
        test_file = pd.read_csv(outte, delimiter=';', header=None)
        test_x = test_file.to_numpy(dtype=int, copy=True).ravel()
        print("text_x:", test_x)
        print("X_dados.shape:", X_dados.shape)
        print("X_dados[0]:", X_dados[0])
        print("X_dados[1]:", X_dados[1])
        print("X_dados:", X_dados)
        X_train_d = X_dados.take(train_x, axis=0)
        print("X_train_d.shape:", X_train_d.shape)
        print("X_train_d[0]:", X_train_d[0])
        print("X_train_d[1]:", X_train_d[1])
        print("X_train_d:", X_train_d)
        data2_df = X_train_d
        y_train_d = y_dados.take(train_x, axis=0)
        X_test_d = X_dados.take(test_x,axis=0)
        y_test_d = y_dados.take(test_x, axis=0)

        X_train, X_test, y_train, y_test = train_test_split(X_train_d, y_train_d, test_size=0.25, random_state=42)
        mean = np.mean(X_train, axis=0)
        std = np.std(X_train, axis=0)
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        lb = LabelEncoder()
        y_train = np_utils.to_categorical(lb.fit_transform(y_train))
        y_test = np_utils.to_categorical(lb.fit_transform(y_test))
        if config["dnn_type"] == "rnn-gru":
            x_traincnn = X_train
            x_testcnn = X_test
        else:
            x_traincnn = np.expand_dims(X_train, axis=2)
            x_testcnn = np.expand_dims(X_test, axis=2)
        model, opt = build_rnn_1gru(x_traincnn)
        file_weights = "tests/models/best_train.h5"
        mcp_save = ModelCheckpoint(file_weights, save_best_only=True, monitor="val_loss", mode="min")
        lr_reduce = ReduceLROnPlateau(monitor="val_loss", patience=5, verbose=2, factor=0.9)
        early_stopping = EarlyStopping(monitor="val_loss", patience=50, verbose=1, mode="min")

        print("Omar-execução fit iniciada", " fold", kfold)
        H = model.fit(x_traincnn, y_train, batch_size=config["batch_size"], epochs=config["max_epochs"],
                      validation_data=(x_testcnn, y_test), callbacks=[mcp_save, lr_reduce, early_stopping])
        score = model.evaluate(x_testcnn, y_test, verbose=0)
        print("Omar-execução fit concluida")
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(np.arange(0, len(H.history["loss"])), H.history["loss"], label="train_loss")
        plt.plot(np.arange(0, len(H.history["loss"])), H.history["val_loss"], label="val_loss")
        plt.plot(np.arange(0, len(H.history["loss"])), H.history["acc"], label="train_acc")
        plt.plot(np.arange(0, len(H.history["loss"])), H.history["val_acc"], label="val_acc")
        plt.xlabel("Epoch_" + str(config["max_epochs"]))
        plt.ylabel("Loss /Accuracy")
        plt.legend(loc="lower left")
        plt.title(f"Training Loss and Accuracy")
        plt.savefig(f"tests/accuracy/graph_train_{aug_type}.png")
        # plt.show()
        file_model = "tests/models/model.json"
        print("Omar-abrindo arquivo model")
        with open(file_model, "w") as json_file:
            json_file.write(model.to_json())
        json_file = open(file_model, "r")
        loaded_model_json = json_file.read()
        json_file.close()
        print("Omar-fechado arquivo model")
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(file_weights)
        print("Omar-execução predict_test dataset validation iniciada")
        y_true_r, y_pred_r = predict_test(x_testcnn, loaded_model, y_test, lb, dataset_type="validation", aug_type=aug_type)
        print("Omar-execução predict_test dataset validation finalizada")

        test_valid = X_test_d
        mean = np.mean(test_valid, axis=0)
        std = np.std(test_valid, axis=0)
        test_valid = (test_valid - mean) / std
        test_valid = np.array(test_valid)
        #  test_valid_lb = np.array(data_df_test.label)
        test_valid_lb = np.array(y_test_d)
        lb_test = LabelEncoder()
        test_valid_lb = np_utils.to_categorical(lb_test.fit_transform(y_test_d))
        if config["dnn_type"] == "rnn-gru":
            test_valid = test_valid
        else:
            test_valid = np.expand_dims(test_valid, axis=2)
        print("Omar-execução predict_test dataset test iniciada")
        y_true_r, y_pred_r = predict_test(test_valid, loaded_model, test_valid_lb, lb_test, dataset_type="test", aug_type=aug_type)
        y_true_r = np.asarray(y_true_r)
        y_pred_r = np.asarray(y_pred_r)
        print("Omar-execução predict_test dataset test finalizada")
        ## calculo da matriz confusão
        fp = []
        vn = []
        fn = []
        vp = []

        for em in EMOTION_NUMBERS:
            fp.append(0)
            vn.append(0)
            fn.append(0)
            vp.append(0)
        print("y_true_r:", y_true_r)
        print("y_true_r[0]:", y_true_r[0])
        print("y_true_r[1]:", y_true_r[1])
        print("type(y_true_r[0]):", type(y_true_r[0]))
        print("len(y_true_r):", len(y_true_r))

        for i in range(0, len(y_true_r)):
            for em in EMOTION_NUMBERS:
                if y_pred_r[i] == str(em):
                    if y_true_r[i] == y_pred_r[i]:
                        vp[em] = vp[em] + 1
                    else:
                        fp[em] = fp[em] + 1
                else:
                    if str(em) == y_true_r[i]:
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
            new_row = [kfold, i, " ", vp[i], vn[i], fn[i], fp[i], precisao, sensibilidade, y_true_r, y_pred_r]
            with open(out2, 'a', newline='') as file:
                writer = csv.writer(file, delimiter=';')
                writer.writerow(new_row)

        em = len(EMOTION_NUMBERS)
        if precisao_m == 0 or sensibilidade_m == 0:
            f1score_m = [data_set, kfold, "f1-scorem=", 0]
        else:
            f1score_m = [data_set, kfold, "f1-scorem=",
                         2 * (precisao_m / em) * (sensibilidade_m / em) / (precisao_m / em + sensibilidade_m / em)]
        with open(out3, 'a', newline='') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow(f1score_m)
    ##############################################################
elif data_set in ("TRTE-BADEM-VERBO", "TRTE-VERBO-BADEM"):
    # print("X_dadostr.shape:", X_dadostr.shape)
    # print("X_dadoste.shape:", X_dadoste.shape)
    X_train_d = X_dadostr
    data2_df = X_train_d
    y_train_d = y_dadostr
    X_test_d = X_dadoste
    y_test_d =y_dadoste

    X_train, X_test, y_train, y_test = train_test_split(X_train_d, y_train_d, test_size=0.25, random_state=42)
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    lb = LabelEncoder()
    y_train = np_utils.to_categorical(lb.fit_transform(y_train))
    y_test = np_utils.to_categorical(lb.fit_transform(y_test))
    if config["dnn_type"] == "rnn-gru":
        x_traincnn = X_train
        x_testcnn = X_test
    else:
        x_traincnn = np.expand_dims(X_train, axis=2)
        x_testcnn = np.expand_dims(X_test, axis=2)
    model, opt = build_rnn_1gru(x_traincnn)
    file_weights = "tests/models/best_train.h5"
    mcp_save = ModelCheckpoint(file_weights, save_best_only=True, monitor="val_loss", mode="min")
    lr_reduce = ReduceLROnPlateau(monitor="val_loss", patience=5, verbose=2, factor=0.9)
    early_stopping = EarlyStopping(monitor="val_loss", patience=50, verbose=1, mode="min")
    print("Omar-execução fit iniciada")
    H = model.fit(x_traincnn, y_train, batch_size=config["batch_size"], epochs=config["max_epochs"],
                  validation_data=(x_testcnn, y_test), callbacks=[mcp_save, lr_reduce, early_stopping])
    score = model.evaluate(x_testcnn, y_test, verbose=0)
    print("Omar-execução fit concluida")
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, len(H.history["loss"])), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, len(H.history["loss"])), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, len(H.history["loss"])), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, len(H.history["loss"])), H.history["val_acc"], label="val_acc")
    plt.xlabel("Epoch_" + str(config["max_epochs"]))
    plt.ylabel("Loss /Accuracy")
    plt.legend(loc="lower left")
    plt.title(f"Training Loss and Accuracy")
    plt.savefig(f"tests/accuracy/graph_train_{aug_type}.png")
    # plt.show()
    file_model = "tests/models/model.json"
    print("Omar-abrindo arquivo model")
    with open(file_model, "w") as json_file:
        json_file.write(model.to_json())
    json_file = open(file_model, "r")
    loaded_model_json = json_file.read()
    json_file.close()
    print("Omar-fechado arquivo model")
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(file_weights)
    print("Omar-execução predict_test dataset validation iniciada")
    y_true_r, y_pred_r = predict_test(x_testcnn, loaded_model, y_test, lb, dataset_type="validation", aug_type=aug_type)
    print("Omar-execução predict_test dataset validation finalizada")

    test_valid = X_test_d
    mean = np.mean(test_valid, axis=0)
    std = np.std(test_valid, axis=0)
    test_valid = (test_valid - mean) / std
    test_valid = np.array(test_valid)
    #  test_valid_lb = np.array(data_df_test.label)
    test_valid_lb = np.array(y_test_d)
    lb_test = LabelEncoder()
    test_valid_lb = np_utils.to_categorical(lb_test.fit_transform(y_test_d))
    if config["dnn_type"] == "rnn-gru":
        test_valid = test_valid
    else:
        test_valid = np.expand_dims(test_valid, axis=2)
    print("Omar-execução predict_test dataset test iniciada")
    y_true_r, y_pred_r = predict_test(test_valid, loaded_model, test_valid_lb, lb_test, dataset_type="test",
                                      aug_type=aug_type)
    y_true_r = np.asarray(y_true_r)
    y_pred_r = np.asarray(y_pred_r)
    print("Omar-execução predict_test dataset test finalizada")
    ## calculo da matriz confusão
    fp = []
    vn = []
    fn = []
    vp = []

    for em in EMOTION_NUMBERS:
        fp.append(0)
        vn.append(0)
        fn.append(0)
        vp.append(0)

    print("y_true_r:", y_true_r)
    print("y_true_r[0]:", y_true_r[0])
    print("y_true_r[1]:", y_true_r[1])
    print("type(y_true_r[0]):", type(y_true_r[0]))
    print("len(y_true_r):", len(y_true_r))

    for i in range(0, len(y_true_r)):
        for em in EMOTION_NUMBERS:
            if y_pred_r[i] == str(em):
                if y_true_r[i] == y_pred_r[i]:
                    vp[em] = vp[em] + 1
                else:
                    fp[em] = fp[em] + 1
            else:
                if str(em) == y_true_r[i]:
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
        new_row = [0, i, " ", vp[i], vn[i], fn[i], fp[i], precisao, sensibilidade, y_true_r, y_pred_r]
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