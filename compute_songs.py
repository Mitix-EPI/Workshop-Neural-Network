import pandas as pd
labels = pd.read_csv('data/train_labels.csv', index_col=0)

import fma
path = fma.get_audio_path(1042)
nb_genres = 0

print(path)

import librosa
import numpy as np
# from tensorflow.keras.utils import to_categorical
import os
import random

def get_genre_songs(genre, limits=1000):
    global labels
    paths = []
    tmp = labels.loc[labels['genre'] == genre]
    indexes = tmp.index.values
    for i in indexes:
        path = fma.get_audio_path(i)
        if (os.path.exists(path)):
            paths.append(path)  
    if (len(paths) < limits):
        return paths
    random_paths = random.choices(paths, k=limits)
    return random_paths

def extract_features_song(f):
    y, _ = librosa.load(f)

    # get Mel-frequency cepstral coefficients
    mfcc = librosa.feature.mfcc(y)
    # normalize values between -1,1 (divide by max)
    mfcc /= np.amax(np.absolute(mfcc))
    test = np.ndarray.flatten(mfcc)
    return test

nb_genres = 0
all_features = []
all_labels = []

GENRES = ['Electronic', 'Experimental', 'Folk', 'Hip-Hop',
        'Instrumental', 'International', 'Pop', 'Rock']
for genre in GENRES:
    sound_files = get_genre_songs(genre, limits=1000) # 100
    print('Processing %d songs in %s genre...' % (len(sound_files), genre))
    if sound_files:
        nb_genres += 1
    for f in sound_files:
        print("\t-> Processing %s..." % f)
        if (not os.path.isdir("computed/" + (f.split('/')[1]))) :
            os.system("mkdir " + "computed/" + (f.split('/')[1]))
        if (not os.path.isdir("computed/" + (f.split('/')[1]) + '/' + f.split('/')[2])) :
            os.system("mkdir " + "computed/" + (f.split('/')[1]) + '/' + f.split('/')[2])
        if (os.path.isfile("computed" + f[4:-4] + ".csv")):
            continue
        try :
            features = extract_features_song(f)
        except :
             continue
        os.system("touch computed" + f[4:-4] + ".csv")
        np.savetxt("computed" + f[4:-4] + ".csv", features, delimiter=",")
        all_features.append(features)
        all_labels.append(genre)
