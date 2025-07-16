# feature extractor with Opensmile
import os

class FeatureExtractor:

    def __init__(self, audio_folder_path, output_folder_path, opensmile_config_path):
        self.audio_folder_path = audio_folder_path
        self.output_folder_path = output_folder_path
        self.opensmile_config_path = opensmile_config_path
        self.__create_if_does_not_exist(output_folder_path)


    def generate_extract_command(self, filepath):
        filename = filepath.split('/')[-1]
        filetype = self.opensmile_config_path.split('/')[-2]
        if filetype == "Prosody":
            saida = "-csvoutput"
        else:
            saida = "-O"
        # print (filetype)
        # print (saida)
        return "SMILExtract -C {0} -I {1} {2} {3}.csv".format(self.opensmile_config_path, filepath, saida,
                                                             self.output_folder_path + filename)
        #return "SMILExtract -C {0} -I {1} -O {2}.csv".format(self.opensmile_config_path, filepath, self.output_folder_path + filename)

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

arqaudio = 'verbo'

CONFIG_FILE = "chroma_fft-Omar_2023_03_04.conf"
CONFIG_DIR = "Chroma/"
CONFIG_DIR_FILE = CONFIG_DIR + CONFIG_FILE
OPENSMILE_CONFIG_FILEPATH = "/Users/omarr/Documents/Omar/Mestrado/opensmile-3.0-win-x64/config/" + CONFIG_DIR_FILE
AUDIO_PATH = '/Users/omarr/Documents/Omar/Mestrado/audio/' + arqaudio
OUTPUT_PATH = '/Users/omarr/Documents/Omar/Mestrado/python/EmotionsThroughAcoustic/datasetsv/Original/' + CONFIG_DIR
extracor = FeatureExtractor(AUDIO_PATH, OUTPUT_PATH, OPENSMILE_CONFIG_FILEPATH)
extracor.extract_features()

CONFIG_FILE = "MFCC_CSV_OUT.conf"
CONFIG_DIR = "MFCC/"
CONFIG_DIR_FILE = CONFIG_DIR + CONFIG_FILE
OPENSMILE_CONFIG_FILEPATH = "/Users/omarr/Documents/Omar/Mestrado/opensmile-3.0-win-x64/config/" + CONFIG_DIR_FILE
AUDIO_PATH = '/Users/omarr/Documents/Omar/Mestrado/audio/' + arqaudio
OUTPUT_PATH = '/Users/omarr/Documents/Omar/Mestrado/python/EmotionsThroughAcoustic/datasetsv/Original/' + CONFIG_DIR
extracor = FeatureExtractor(AUDIO_PATH, OUTPUT_PATH, OPENSMILE_CONFIG_FILEPATH)
extracor.extract_features()

CONFIG_FILE = "prosodyAcf-omar.conf"
CONFIG_DIR = "Prosody/"
CONFIG_DIR_FILE = CONFIG_DIR + CONFIG_FILE
OPENSMILE_CONFIG_FILEPATH = "/Users/omarr/Documents/Omar/Mestrado/opensmile-3.0-win-x64/config/" + CONFIG_DIR_FILE
AUDIO_PATH = '/Users/omarr/Documents/Omar/Mestrado/audio/' + arqaudio
OUTPUT_PATH = '/Users/omarr/Documents/Omar/Mestrado/python/EmotionsThroughAcoustic/datasetsv/Original/' + CONFIG_DIR
extracor = FeatureExtractor(AUDIO_PATH, OUTPUT_PATH, OPENSMILE_CONFIG_FILEPATH)
extracor.extract_features()

arqaudio = 'badem'

CONFIG_FILE = "chroma_fft-Omar_2023_03_04.conf"
CONFIG_DIR = "Chroma/"
CONFIG_DIR_FILE = CONFIG_DIR + CONFIG_FILE
OPENSMILE_CONFIG_FILEPATH = "/Users/omarr/Documents/Omar/Mestrado/opensmile-3.0-win-x64/config/" + CONFIG_DIR_FILE
AUDIO_PATH = '/Users/omarr/Documents/Omar/Mestrado/audio/' + arqaudio
OUTPUT_PATH = '/Users/omarr/Documents/Omar/Mestrado/python/EmotionsThroughAcoustic/datasets/Original/' + CONFIG_DIR
extracor = FeatureExtractor(AUDIO_PATH, OUTPUT_PATH, OPENSMILE_CONFIG_FILEPATH)
extracor.extract_features()

CONFIG_FILE = "MFCC_CSV_OUT.conf"
CONFIG_DIR = "MFCC/"
CONFIG_DIR_FILE = CONFIG_DIR + CONFIG_FILE
OPENSMILE_CONFIG_FILEPATH = "/Users/omarr/Documents/Omar/Mestrado/opensmile-3.0-win-x64/config/" + CONFIG_DIR_FILE
AUDIO_PATH = '/Users/omarr/Documents/Omar/Mestrado/audio/' + arqaudio
OUTPUT_PATH = '/Users/omarr/Documents/Omar/Mestrado/python/EmotionsThroughAcoustic/datasets/Original/' + CONFIG_DIR
extracor = FeatureExtractor(AUDIO_PATH, OUTPUT_PATH, OPENSMILE_CONFIG_FILEPATH)
extracor.extract_features()

CONFIG_FILE = "prosodyAcf-omar.conf"
CONFIG_DIR = "Prosody/"
CONFIG_DIR_FILE = CONFIG_DIR + CONFIG_FILE
OPENSMILE_CONFIG_FILEPATH = "/Users/omarr/Documents/Omar/Mestrado/opensmile-3.0-win-x64/config/" + CONFIG_DIR_FILE
AUDIO_PATH = '/Users/omarr/Documents/Omar/Mestrado/audio/' + arqaudio
OUTPUT_PATH = '/Users/omarr/Documents/Omar/Mestrado/python/EmotionsThroughAcoustic/datasets/Original/' + CONFIG_DIR
extracor = FeatureExtractor(AUDIO_PATH, OUTPUT_PATH, OPENSMILE_CONFIG_FILEPATH)
extracor.extract_features()