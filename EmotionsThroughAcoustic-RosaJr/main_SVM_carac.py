## main_SVM_carac.py: feature extraction for mann whitney testing.

import argparse
import os
# import sys
import time
from collections import OrderedDict
import csv
# from pyAudioAnalysis import audioTrainTest as aT
# Alterado Omar - 17/12/2022
from pyAudioAnalysis import audioTrainTest_omar as aT
# Alterado Omar - fim
# Alterado Omar - 09/12/2023
from pyAudioAnalysis import MidTermFeatures_omar as aF
# Alterado Omar - fim
# incluido Omar - 09/12/2023
import numpy as np
# sys.path.append("C:\dev\emotion_speech_recognition\src\dependencies\pyAudioAnalysis")


def print_parameters(st_w, st_s, mt_w, mt_s, perc_train, use_svm, model_name):
    print("\n =============== {} ================ ".format("SVM" if use_svm else "KNN"))
    print(" Parameters for model ’{}’".format(model_name))
    print(" Short term window : {}".format(st_w))
    print(" Short term step : {}".format(st_s))
    print("Mid term window : {}".format(mt_w))
    print("Mid term step : {}".format(mt_s))
    print(" Utilizing {:.0f}% of samples for training ".format(perc_train * 100))
    print(" ==================================== ")


def feature_and_train(samples_prefix, st_w, st_s, mt_w, mt_s, perc_train, confusion_matrix_perc, use_svm, 
                      verbosity, model_name):
    start = time.time()
    list_of_dirs_or_classes = []
    dirs = [d for d in os.listdir(samples_prefix) if os.path.isdir(os.path.join(samples_prefix, d))]
    for dire in dirs:
        list_of_dirs_or_classes.append(samples_prefix + str(dire))
    print(" Starting training ...")
    if use_svm:
        features, class_names, file_names, feature_names_arr = \
            aF.multiple_directory_feature_extraction(list_of_dirs_or_classes, mt_w, mt_s, st_w, st_s,
                                                     compute_beat=False)
        outx = "data_setx" + "_" + "carac.csv"
        carac_vetor = []
        for i in range(len(class_names)):
            for j in range(len(file_names[i])):
                new_row = np.concatenate((class_names[i], file_names[i][j]), axis=None)
                for k in range(len(features[i][j])):
                    new_row = np.concatenate((new_row, features[i][j][k]), axis=None)
                carac_vetor.append(new_row)
                with open(outx, 'a', newline='') as file:
                    writer = csv.writer(file, delimiter=';')
                    writer.writerow(new_row)
        model_name = "models/SVM_ " + model_name
        print_parameters(st_w, st_s, mt_w, mt_s, perc_train, use_svm, model_name)

        best_perf = 0
        end = time.time()
        print(" Finished in {:10.2f} seconds ".format(end - start))
        # Alterado Omar - 17/12/2022
        return carac_vetor
        # Alterado Omar - fim

def test_model(prefix, model, tfold, tfilename):
    dirs = os.listdir(prefix)
    use_svm = True
    model = ("models/SVM_ " + model) if use_svm else ("models/KNN_" + model)
    print("\n Testing model : {}".format(model))
    print(" Confusion Matrix :")
    for d in dirs:
        print("\t{}".format(d[:4]))
        print(" ")
    total_correct = 0
    total_files = 0
    # INCLUIDO OMAR - 06/08/2023
    total_class_vp = OrderedDict()
    total_class_fp = OrderedDict()
    total_class_fn = OrderedDict()
    total_class_vn = OrderedDict()
    for dirx in dirs:
        total_class_vp[dirx] = 0
        total_class_fp[dirx] = 0
        total_class_fn[dirx] = 0
        total_class_vn[dirx] = 0
    # FIM INCLUIDO OMAR - 06/08/2023

    for classs in dirs:
        files = os.listdir(prefix + classs)
        print("files: ", files)
        print("prefix: ", prefix)
        print("classs: ", classs) 
        class_files_num = len(files)
        total_files = total_files + class_files_num
        classified = OrderedDict()
        for dirx in dirs:
            classified[dirx] = 0
            print("{}\t".format(classs[:4]))
        for file in files:
            file_to_test = prefix + classs + "/" + file
            class_classified = test_file(file_to_test, model, use_svm)
            classified[class_classified] = classified[class_classified] + 1
            if class_classified == classs:
                total_correct = total_correct + 1
                total_class_vp[classs] = total_class_vp[classs] + 1
            else:
                total_class_fp[class_classified] = total_class_fp[class_classified] + 1
                total_class_fn[classs] = total_class_fn[classs] + 1
                for ix in dirs:
                    if ix not in (class_classified, classs):
                        total_class_vn[ix] = total_class_vn[ix] + 1
        for key in classified.keys():
            print("{}\t".format(round((classified[key] / float(class_files_num)) * 100, 2)))
            print(" ")
            print(" General accuracy is {}%".format(round(total_correct /
                                                          float(total_files) * 100, 2)))
    print("Totais (vp,fp,fn,vn): ", total_class_vp, total_class_fp, total_class_fn, total_class_vn)
    kfold = tfold
    data_set = tfilename
    out1 = data_set + "-" + "output1.csv"

    for dirx in dirs:
        if total_class_vp[dirx] == 0:
            precisao = 0
            sensibilidade = 0
        else:
            precisao = total_class_vp[dirx] / (total_class_vp[dirx] + total_class_fp[dirx])
            sensibilidade = total_class_vp[dirx] / (total_class_vp[dirx] + total_class_fn[dirx])

        new_row = [kfold, dirx,
                   total_class_vp[dirx],
                   total_class_fp[dirx],
                   total_class_vn[dirx],
                   total_class_fn[dirx],
                   precisao,
                   sensibilidade
                   ]
        with open(out1, 'a', newline='') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow(new_row)


def test_file(filename_to_test, model_name, use_svm=True, verbose=False):
    # if os.path.isfile(filename_to_test):
    if True is True:
        start = time.time()
        if use_svm:
            r, P, classNames = aT.file_classification(filename_to_test, model_name, "svm")
        else:
            r, P, classNames = aT.file_classification(filename_to_test, model_name, "svm")
        chosen = 0.0
        chosenClass = ""
        print("r:", r)
        print("P:", P)
        print("classNames:", classNames)
        if len(P) == len(classNames):
            for i in range(0, len(P), 1):
                if P[i] > chosen:
                    chosen = P[i]
                    chosenClass = classNames[i]
        end = time.time()
        if verbose:
            # print("\n \n The audio file was classified as {} with prob {}% in {:.2 f} seconds \n\n". \
            #      format(chosenClass.upper(), round(chosen * 100, 2), end - start))
            print("\n \n The audio file was classified as {} with prob {}% in {:.2f} seconds \n \n".
                  format(chosenClass.upper(), round(chosen * 100, 2), end - start))
        return chosenClass
    else:
        print(os.path.isfile(filename_to_test))
        print(os.path.dirname(os.path.abspath(" ")) + "/" + filename_to_test)
        print(" File doesnt exists : {}".format(filename_to_test))

    return None


def train_until_get_better_acc(samples_prefix, model_name, train_until):
    best_acc = 0
    for i in range(0, 1000):
        accuracy = train_SVM(samples_prefix, model_name)
        print("\n Current accuracy : {}".format(accuracy))
        print(" Best accuracy : {}\n\n".format(best_acc))
        if accuracy > best_acc:
            prefix = samples_prefix + " ../../ src / models /"
            os.rename(prefix + " SVM_port_single ", prefix + "best_SVM_port_single ")
            os.rename(prefix + " SVM_port_single . arff ", prefix + "best_SVM_port_single . arff ")
            os.rename(prefix + " SVM_port_singleMEANS ", prefix + "best_SVM_port_singleMEANS ")
            best_acc = accuracy
            if accuracy >= train_until:
                break


def train_SVM(samples_prefix, model_name):
    # ======= PORTUGUESE BEST CONFIGURATION ========
    SHORT_TERM_WINDOW = 0.1
    SHORT_TERM_STEP = 0.033
    MID_TERM_WINDOW = 1.2
    MID_TERM_STEP = 0.6
    confusion_matrix_perc = True
    use_svm = True
    perc_train = 0.75
    VERBOSITY = False
    return feature_and_train(samples_prefix, SHORT_TERM_WINDOW,
                             SHORT_TERM_STEP, MID_TERM_WINDOW, MID_TERM_STEP,
                             perc_train,
                             confusion_matrix_perc, use_svm, VERBOSITY, model_name)


def train_KNN(samples_prefix, model_name):
    SHORT_TERM_WINDOW = 0.036
    SHORT_TERM_STEP = 0.012
    MID_TERM_WINDOW = 1.3
    MID_TERM_STEP = 0.65
    confusion_matrix_perc = True
    use_svm = False
    perc_train = 0.75
    VERBOSITY = False
    feature_and_train(samples_prefix, SHORT_TERM_WINDOW,
                      SHORT_TERM_STEP, MID_TERM_WINDOW, MID_TERM_STEP,
                      perc_train,
                      confusion_matrix_perc, use_svm, VERBOSITY, model_name)


def brute_force_training(samples_prefix):
    min_st = 0.020
    max_st = 0.100
    step_st = 0.001
    st_overl = 0.33
    min_mt = 1.000
    max_mt = 3.000
    step_mt = 0.100
    mt_overl = 0.5
    MID_TERM_WINDOW = min_mt
    MID_TERM_STEP = round(MID_TERM_WINDOW * mt_overl, 3)
    SHORT_TERM_WINDOW = min_st
    SHORT_TERM_STEP = round(SHORT_TERM_WINDOW * st_overl, 3)
    confusion_matrix_perc = True
    use_svm = True
    perc_train = 0.75
    VERBOSITY = False
    bestAcc = 0.0
    bestAccParams = {" st_w ": SHORT_TERM_WINDOW,
                     " st_s ": SHORT_TERM_STEP,
                     " mt_w ": MID_TERM_WINDOW,
                     " mt_s ": MID_TERM_STEP}
    range_mt_max = int(round(max_mt - min_mt, 3) / step_mt) + 1
    range_st_max = int(round(max_st - min_st, 3) / step_st) + 1
    for mt in range(0, range_mt_max):
        SHORT_TERM_WINDOW = min_st
        SHORT_TERM_STEP = round(SHORT_TERM_WINDOW * st_overl, 3)
        for st in range(0, range_st_max):
            accuracy = feature_and_train(samples_prefix,
                                         SHORT_TERM_WINDOW, SHORT_TERM_STEP, MID_TERM_WINDOW,
                                         MID_TERM_STEP, perc_train,
                                         confusion_matrix_perc, use_svm, VERBOSITY,
                                         " brute_force_test_model ")
            SHORT_TERM_WINDOW = SHORT_TERM_WINDOW + step_st
            SHORT_TERM_STEP = round(SHORT_TERM_WINDOW * st_overl, 3)
            if accuracy > bestAcc:
                bestAcc = accuracy
                bestAccParams[" st_w "] = SHORT_TERM_WINDOW
                bestAccParams[" st_s "] = SHORT_TERM_STEP
                bestAccParams[" mt_w "] = MID_TERM_WINDOW
                bestAccParams[" mt_s "] = MID_TERM_STEP
        MID_TERM_WINDOW = MID_TERM_WINDOW + step_mt
        MID_TERM_STEP = round(MID_TERM_WINDOW * mt_overl, 3)
        print("\n \n Melhor precisao : {}".format(bestAcc))
        print(" SHORT_TERM_WINDOW = {} \n SHORT_TERM_STEP = {} \n MID_TERM_WINDOW = {} \n "
              "MID_TERM_STEP = {}\n\n".format(bestAccParams[" st_w "], bestAccParams[" st_s "],
                                              bestAccParams[" mt_w "], bestAccParams[" mt_s "]))

def main_SVM_carac(filename, fold):
    print("__name__: ", __name__)
    # if __name__ == "__main__":
    PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
    print("project_path:", PROJECT_PATH)
    print("os.listdir:", os.listdir(PROJECT_PATH))
    print("os.path.dirname:", os.path.dirname(PROJECT_PATH))
    parser = argparse.ArgumentParser()
    print("passo 1")
    SAMPLES_PREFIX = PROJECT_PATH + "/train/"
    TEST_PREFIX = PROJECT_PATH + "/test/"
    MODEL_NAME = filename + str(fold)
    print("model_name=", MODEL_NAME)
    return(train_SVM(SAMPLES_PREFIX, MODEL_NAME))