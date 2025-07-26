# Emotions-Through-Acoustic
Article title: Performance evaluation of emotion recognition algorithms in Brazilian Portuguese language audios

Journal: IEEE Latin American Transactions

Submission ID 9577

Description: This project makes a performance evaluation of three algorithms which make a emotion recognition using acoustic features: Deep, RosaJr and Souza. Each folder contains the code used to evaluate the performance for each algorithm. 

Authors - affiliation:

Omar Rodrigues da Silva - Universidade de São Paulo - Escola de Artes, Ciências e Humanidades

Luisa Medina Fermino Carlos - Paradigma Center for Behavioral Science and Technology

Felipe Corchs - Universidade de São Paulo - Faculdade de Medicina

Ariane Machado-Lima - Universidade de São Paulo - Escola de Artes, Ciências e Humanidades

Fatima L. S. Nunes - Universidade de São Paulo - Escola de Artes, Ciências e Humanidades 

Datasets:

VERBO: request this dataset through email to jrtorresneto@gmail.com

BADEM: request this dataset through email to datasetemocoes.badem@gmail.com

Code:

folder EmotionsThroughAcoustic-Deep: contains the Python code to evaluate the performance of algorithm DEEP and the Opensmile files to extract the audio features (*.conf). 

- "Dataset_caracterizacao_mann.py": Whitney-Mann testing of features used on Deep algorithm for emotion recognition
- "Generate train-test-files.py": generate the indexes of each fold for testing on cross-validation.
- "Train and test emotions-portuguese.py": routine to train and test for each type of testing
- "conversor mp4-wave.py": convert audios from .mp4 to .wav (this is used for all algorithms)
- "dataset_loader.py": routine to load BADEM dataset features
- "dataset_loaderv.py": routine to load VERBO dataset features
- "feature_extractor.py": feature extrator using OPENSmile

folder EmotionsThroughAcoustic-RosaJr: contains the Python code to evaluate the performance of algorithm RosaJr
- "Copiar train-test-files.py": routine to call the train and test for each type of testing
- "audioTrainTest_Omar.py": copy of routine auditoTrainTest.py" (PyAudioanalysis) to run under windows (to include the windows operating system separator for files)
- "correcao audio-codec-freq.py": adjustments of audio features coded and frequency to standardize them (the audios have different codec and frequency)
- "dataset_loader.py": routine to load BADEM dataset features
- "dataset_loaderv.py": routine to load VERBO dataset features
- "geracao caracteristicas.py": Whitney-Mann testing of features used on RosaJr algorithm for emotion recognition
- "mainSVM.py": routine to train and test SVM using PyAudioanalysis
- "mainSVM_carac.py": routine to extract the features to be used in the Whitney-Mann testing (called by geracao caracteristicas.py)

folder EmotionsThroughAcoustic-Souza: contains the Python code to evaluate the performance of algorithm Souza
- "Chamada_souza.py": routine to train and test for each type of testing
- "Chamada_souza_mann.py": Whitney-Mann testing of features used on Souza algorithm for emotion recognition

