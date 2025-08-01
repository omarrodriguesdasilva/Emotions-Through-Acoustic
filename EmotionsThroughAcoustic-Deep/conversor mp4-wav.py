# conversion of audio files from mp4 to wav. 
import os
AUDIO_PATH = '/Users/omarr/Documents/Omar/Mestrado/python/EmotionsThroughAcoustic/DatasetLuisa/000'
OUTPUT_PATH = 'DatasetLuisa/wav/'

class FeatureExtractor:

    def __init__(self, audio_folder_path, output_folder_path):
        self.audio_folder_path = audio_folder_path
        self.output_folder_path = output_folder_path
        self.__create_if_does_not_exist(output_folder_path)


    def generate_extract_command(self, filepath):
        filename1 = filepath.split('/')[-1]
        filename = filename1.split('.')[0]
        return "ffmpeg -i {0} {1}.wav".format(filepath, self.output_folder_path + filename)

    def get_all_filepaths(self, path):
        result_filepaths = []
        for inst in os.listdir(path):
            recursive_file_instances = []
            if os.path.isdir("{}/{}".format(path, inst)):
                recursive_file_instances = self.get_all_filepaths("{}/{}".format(path, inst))
                for filepath in recursive_file_instances:
                    result_filepaths.append(filepath)
            else:
                result_filepaths.append("{}/{}".format(path, inst))

        return result_filepaths + recursive_file_instances

    def extract_features(self):
        filepaths = self.get_all_filepaths(self.audio_folder_path)
        for filepath in filepaths:
            extract_command = self.generate_extract_command(filepath)
            print(extract_command)
            os.system(extract_command)

    @staticmethod
    def __create_if_does_not_exist(audio_folder_output):
        if not os.path.exists(audio_folder_output):
            os.makedirs(audio_folder_output)


extracor = FeatureExtractor(AUDIO_PATH, OUTPUT_PATH)
extracor.extract_features()
