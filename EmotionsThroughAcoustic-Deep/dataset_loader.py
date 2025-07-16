##
## dataset_loader.py - load dataset VERBO
##

import os

import pandas as pd
import numpy as np

class DatasetLoader:

    def __init__(self, dataset_path=None):
        self.dataset_path = dataset_path

    def compose_complete_dataset(self):
        # TODO: set this as an argument
        DATASET_PATH = 'datasets/Original/' \
                       '/'
        dataset_loader = DatasetLoader(DATASET_PATH)
        mfcc_features, y = dataset_loader.get_dataset()

        DATASET_PATH = 'datasets/Original/Prosody/'
        dataset_loader = DatasetLoader(DATASET_PATH)
        prosody_features, y = dataset_loader.get_dataset()

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

        DATASET_PATH = 'datasets/Original/Chroma/'
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

        X = np.asarray(new_dataset)
        return X, y

    def translate_emotion(self, file_path, type_='default'):
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

    def get_all_filepaths(self, path):
        result_filepaths = []
        for inst in os.listdir(path):
            recursive_file_instances = []
            if os.path.isdir("{}/{}".format(self.dataset_path, inst)):
                recursive_file_instances = self.get_all_filepaths("{}/{}".format(path, inst))
                for filepath in recursive_file_instances:
                    result_filepaths.append(filepath)

            else:
                result_filepaths.append("{}/{}".format(path, inst))

        return result_filepaths + recursive_file_instances

    def get_dataset(self):
        X_dataset = []
        Y_dataset = []
        file_paths = self.get_all_filepaths(self.dataset_path)
        #print("get_dataset:", file_paths)
        for file_path in file_paths:
            try:
                actor, gender, filename = self.get_extra_info(file_path)
                inst = pd.read_csv(file_path, delimiter=';')
                if self.dataset_path in ("datasets/Original/Prosody"):
                    inst2 = inst.iloc[0:, 3:]
                else:
                    inst2 = inst.iloc[0:, 2:]
                X_dataset.append(inst2.values)
                # Fim - Alteração Omar.
                Y_dataset.append([self.translate_emotion(file_path), actor, filename])
            except:
                pass
        #print("get_dataset2:", X_dataset)
        #print(Y_dataset)
        return X_dataset, Y_dataset
  
    def get_extra_info(self, file_path):
        #tab_actor=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        tab_gen=['m','m','f','f','f','f','7','m','f','m','11','m','m','f']
        filename = file_path.split('/')[-1]
        actor = filename.split('_')[1]
        #print(filename)
        gender = tab_gen[int(actor)-1]

        return actor, gender, filename

    @staticmethod
    def get_genre(file_path):
        file_name = file_path.split('/')[-1]
        token = file_name.split('-')[1]
        if token[0] == 'f':
            return 'F'
        else:
            return 'M'

    def get_emotion(self, file_path):
        filename = file_path.split('/')[-1]
        emotion_token = filename.split('-')[0]
        return emotion_token