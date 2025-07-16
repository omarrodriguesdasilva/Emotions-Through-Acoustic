# correcao audio-codec-freq.py: fix the codec and frequency from audios to standardize
import os

AUDIO_PATH  = '/Users/omarr/Documents/Omar/Mestrado/python/EmotionsThroughAcoustic-RosaJr/Seriesemocionais'
OUTPUT_PATH = '/Users/omarr/Documents/Omar/Mestrado/python/EmotionsThroughAcoustic-RosaJr/Seriesemocionaiscorr'
filereplace1 = "Seriesemocionais"
filereplace2 = "Seriesemocionaiscorr"

class FeatureExtractor:

    def __init__(self, audio_folder_path, output_folder_path):
        self.audio_folder_path = audio_folder_path
        self.output_folder_path = output_folder_path
        self.__create_if_does_not_exist(output_folder_path)

    def generate_extract_command(self, filepath):
        filename1 = filepath.split('/')[-1]
        filename = filename1.split('.')[0]
        # outfilename = filepath.replace("audio", "audiocorr")
        outfilename = filepath.replace(filereplace1, filereplace2)
        outfilename = outfilename.replace("/", "\\")
        filepath = filepath.replace("/", "\\")
        return "ffmpeg -i {0} -ar 44100 -ac 1 -acodec pcm_s16le {1}".format(filepath, outfilename)

    def get_all_filepaths(self, path):
        result_filepaths = []
        fileoutdir = path.replace("/", "\\")
        fileoutdir = fileoutdir.replace(filereplace1, filereplace2)
        print("fileoutdir:", fileoutdir)
        print("os.path.exists(fileoutdir):", os.path.exists(fileoutdir))
        if not os.path.exists(fileoutdir):
            os.makedirs(fileoutdir)
            print("makedirs", fileoutdir)
        for inst in os.listdir(path):
            recursive_file_instances = []
            print("inst:", inst)
            if os.path.isdir("{}/{}".format(path, inst)):
                recursive_file_instances = self.get_all_filepaths("{}/{}".format(path, inst))
                for filepath in recursive_file_instances:
                    result_filepaths.append(filepath)
            else:
                result_filepaths.append("{}/{}".format(path, inst))

        return result_filepaths

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
