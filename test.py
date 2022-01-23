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
    test = np.ndarray.flatten(mfcc)[:10000]
    return test

def generate_features_and_labels():
    global nb_genres
    all_features = []
    all_labels = []

    GENRES = ['Blues', 'Classical', 'Country', 'Easy Listening', 'Electronic',
           'Experimental', 'Folk', 'Hip-Hop', 'Instrumental', 'International',
           'Jazz', 'Old-Time / Historic', 'Pop', 'Rock', 'Soul-RnB', 'Spoken']
    for genre in GENRES:
        sound_files = get_genre_songs(genre, limits=5) # 100
        print('Processing %d songs in %s genre...' % (len(sound_files), genre))
        if sound_files:
            nb_genres += 1
        for f in sound_files:
            print("\t-> Processing %s..." % f)
            features = extract_features_song(f)
            all_features.append(features)
            all_labels.append(genre)

    # convert labels to one-hot encoding
    label_uniq_ids, label_row_ids = np.unique(all_labels, return_inverse=True)
    label_row_ids = label_row_ids.astype(np.int32, copy=False)
    # onehot_labels = to_categorical(label_row_ids, len(label_uniq_ids))
    onehot_labels = np.eye(len(label_uniq_ids))[label_row_ids]
    return np.stack(all_features), onehot_labels

features, labels = generate_features_and_labels()

print("np.shape(features): ", np.shape(features))
print("np.shape(labels): ", np.shape(labels))

# import tensorflow as tf
# from tensorflow.keras import layers
# from tensorflow.keras import activations
# from keras.models import Sequential
# from keras.layers import Dense, Activation

training_split = 0.8

# last column has genre, turn it into unique ids
alldata = np.column_stack((features, labels))

np.random.shuffle(alldata)
splitidx = int(len(alldata) * training_split)
train, test = alldata[:splitidx,:], alldata[splitidx:,:]

print("np.shape(train): ", np.shape(train))
print("np.shape(test): ", np.shape(test))

print("nb_genres: ", nb_genres)

train_input = train[:,:-nb_genres]
train_labels = train[:,-nb_genres:].astype(int)

test_input = test[:,:-nb_genres]
test_labels = test[:,-nb_genres:].astype(int)

print("np.shape(train_input): ", np.shape(train_input))
print("np.shape(train_labels): ", np.shape(train_labels))

from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,2), random_state=1)
clf.fit(train_input, train_labels)

predict_test = clf.predict(test_input)
print("predict_test: ", predict_test)
print("test_labels: ", test_labels)

from sklearn.metrics import classification_report,confusion_matrix

print(confusion_matrix(test_labels.argmax(axis=1),predict_test.argmax(axis=1)))
print(classification_report(test_labels.argmax(axis=1),predict_test.argmax(axis=1)))


# model = tf.keras.Sequential([
#     layers.Dense(100, input_dim=np.shape(train_input)[1]),
#     layers.Activation(activations.relu),
#     layers.Dense(10),
#     layers.Activation(activations.softmax),
#     ])

# model.compile(optimizer='adam',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])
# print(model.summary())

# model.fit(train_input, train_labels, epochs=10, batch_size=32,
#           validation_split=0.2)

# loss, acc = model.evaluate(test_input, test_labels, batch_size=32)

# print("Done!")
# print("Loss: %.4f, accuracy: %.4f" % (loss, acc))

# TODO

# Rework NN -> Not working not very good

# NEXT STEPS (3rd step)

# How to upgrade the NN ?
    # -> Create a cross-validation set.
    # -> Plot learning curves. (https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html)
    # -> Establish if there is high bias or high variance.
    # -> Tune hyperparameters.
