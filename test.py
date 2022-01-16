import pandas as pd
labels = pd.read_csv('data/train_labels.csv', index_col=0)

import fma
path = fma.get_audio_path(1042)

print(path)

import librosa
import numpy as np
from tensorflow.keras.utils import to_categorical
import os
import random

# function to get all the tracks from a genre in labels variable
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

    return np.ndarray.flatten(mfcc)[:25000]

def generate_features_and_labels():
    all_features = []
    all_labels = []

    GENRES = ['Blues', 'Classical', 'Country', 'Easy Listening', 'Electronic',
           'Experimental', 'Folk', 'Hip-Hop', 'Instrumental', 'International',
           'Jazz', 'Old-Time / Historic', 'Pop', 'Rock', 'Soul-RnB', 'Spoken']
    for genre in GENRES:
        sound_files = get_genre_songs(genre, limits=50)
        print('Processing %d songs in %s genre...' % (len(sound_files), genre))
        for f in sound_files:
            print("\t-> Processing %s..." % f)
            features = extract_features_song(f)
            all_features.append(features)
            all_labels.append(genre)

    # convert labels to one-hot encoding
    label_uniq_ids, label_row_ids = np.unique(all_labels, return_inverse=True)
    label_row_ids = label_row_ids.astype(np.int32, copy=False)
    onehot_labels = to_categorical(label_row_ids, len(label_uniq_ids))
    return np.stack(all_features), onehot_labels

features, labels = generate_features_and_labels()

print(np.shape(features))
print(np.shape(labels))

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import activations
# from keras.models import Sequential
# from keras.layers import Dense, Activation

training_split = 0.8

# last column has genre, turn it into unique ids
alldata = np.column_stack((features, labels))

np.random.shuffle(alldata)
splitidx = int(len(alldata) * training_split)
train, test = alldata[:splitidx,:], alldata[splitidx:,:]

print(np.shape(train))
print(np.shape(test))

train_input = train[:,:-10]
train_labels = train[:,-10:]

test_input = test[:,:-10]
test_labels = test[:,-10:]

print(np.shape(train_input))
print(np.shape(train_labels))


model = tf.keras.Sequential([
    layers.Dense(100, input_dim=np.shape(train_input)[1]),
    layers.Activation(activations.relu),
    layers.Dense(10),
    layers.Activation(activations.softmax),
    ])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
print(model.summary())

model.fit(train_input, train_labels, epochs=10, batch_size=32,
          validation_split=0.2)

loss, acc = model.evaluate(test_input, test_labels, batch_size=32)

print("Done!")
print("Loss: %.4f, accuracy: %.4f" % (loss, acc))

# TODO

# Rework NN -> Not working not very good

# NEXT STEPS (3rd step)

# How to upgrade the NN ?
    # -> Create a cross-validation set.
    # -> Plot learning curves. (https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html)
    # -> Establish if there is high bias or high variance.
    # -> Tune hyperparameters.
