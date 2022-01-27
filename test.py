import pandas as pd
import time
labels = pd.read_csv('./train_labels.csv', index_col=0)

import fma
nb_genres = 0
nb_features = 10000

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

def time_convert(sec):
    mins = sec // 60
    sec = sec % 60
    hours = mins // 60
    mins = mins % 60
    print("\nTime Lapsed = {0}:{1}:{2}\n".format(int(hours),int(mins),sec))

def get_features_song(f):
    global nb_features
    try:
        features = np.genfromtxt(f, delimiter=',')[:nb_features]
        if (len(features) == nb_features):
            return features
        return []
    except:
        return []
    
def display_details_compute(genres, arr_nb_songs_by_genre):
    for i in range(len(genres)):
        print("{} songs in {} genre".format(arr_nb_songs_by_genre[i], genres[i]))

def generate_features_and_labels():
    global nb_genres
    all_features = []
    all_labels = []

    GENRES = ['Electronic', 'Experimental', 'Folk', 'Hip-Hop',
            'Instrumental', 'International', 'Pop', 'Rock']
    arr_nb_songs_by_genre = []
    start_time = time.time() # Calc time to compute
    for genre in GENRES:
        songs_computed = 0
        sound_files = get_genre_songs(genre, limits=5) # 100
        print('Processing %d songs in %s genre...' % (len(sound_files), genre))
        if sound_files:
            nb_genres += 1
        for f in sound_files:
            if (not os.path.isfile(f)) :
                continue
            print("\t-> Processing %s ..." % f)
            features = get_features_song(f)
            if len(features):
                all_features.append(features)
                all_labels.append(genre)
                songs_computed += 1
        arr_nb_songs_by_genre.append(songs_computed)
    # convert labels to one-hot encoding
    label_uniq_ids, label_row_ids = np.unique(all_labels, return_inverse=True)
    label_row_ids = label_row_ids.astype(np.int32, copy=False)
    # onehot_labels = to_categorical(label_row_ids, len(label_uniq_ids))
    onehot_labels = np.eye(len(label_uniq_ids))[label_row_ids]
    end_time = time.time()
    time_lapsed = end_time - start_time
    time_convert(time_lapsed) # Show time to compute
    print(min([np.shape(i) for i in all_features]))
    print(display_details_compute(GENRES, arr_nb_songs_by_genre))
    return np.stack(all_features), onehot_labels

features, labels = generate_features_and_labels()

print("np.shape(features): ", np.shape(features))
print("np.shape(labels): ", np.shape(labels))

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

start_time = time.time() # Calc time to compute

nb_hidden_layer_sizes = (15,)

print("\nCreating model...")
clf = MLPClassifier(solver='lbfgs', alpha=1e-10, hidden_layer_sizes=nb_hidden_layer_sizes, activation = 'logistic', random_state=5, max_iter=15000, learning_rate_init = 0.1)
print("Training the model...")
clf.fit(train_input, train_labels)

end_time = time.time()
time_lapsed = end_time - start_time
time_convert(time_lapsed) # Show time to compute

print("Testing Neural Network...")
predict_test = clf.predict(test_input)

from sklearn.metrics import classification_report,confusion_matrix

print(confusion_matrix(test_labels.argmax(axis=1),predict_test.argmax(axis=1)))
print(classification_report(test_labels.argmax(axis=1),predict_test.argmax(axis=1)))

ans = input("Do you want to generate the NN graph [Y/N] ? ")
if (ans == "Y" or ans == "y" or ans == ""):
    import matplotlib.pyplot as plt
    from draw_neural_net import draw_neural_net

    print("Generating Neural Network Graph ...")
    fig = plt.figure(figsize=(12, 12))
    ax = fig.gca()
    ax.axis('off')

    layer_sizes = [int(nb_features / 1000)] + list(nb_hidden_layer_sizes) + [nb_genres]
    draw_neural_net(ax, .1, .9, .1, .9, layer_sizes, clf.coefs_, clf.intercepts_, clf.n_iter_, clf.loss_)
    fig.savefig('nn_digaram.png')

# TODO

# How to upgrade the NN ?
    # -> Create a cross-validation set.
    # -> Plot learning curves. (https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html)
    # -> Establish if there is high bias or high variance.
    # -> Tune hyperparameters.
