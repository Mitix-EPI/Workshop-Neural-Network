import warnings
import pandas as pd
import matplotlib.pyplot as plt
import time
from scipy import rand
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn import datasets
from sklearn.exceptions import ConvergenceWarning
import numpy as np
import os
import random
import fma

genre_labels = pd.read_csv('./computed/train_labels.csv', index_col=0)
nb_genres = 0
nb_features = 10000 # Variable that you can change
nb_music_by_genre = 5 # Varaible that you can change

# different learning rate schedules and momentum parameters
params = [
    {
        "solver": "sgd",
        "alpha" : 1e-10,
        "activation": 'logistic',
        "random_state" : 5,
        "max_iter": 15000,
        "learning_rate_init": 0.1,
        "hidden_layer_sizes" : (5,)
    },
    {
        "solver": "sgd",
        "alpha" : 1e-10,
        "activation": 'logistic',
        "random_state" : 5,
        "max_iter": 15000,
        "learning_rate_init": 0.1,
        "hidden_layer_sizes" : (10,)
    },
    {
        "solver": "sgd",
        "alpha" : 1e-10,
        "activation": 'logistic',
        "random_state" : 5,
        "max_iter": 15000,
        "learning_rate_init": 0.1,
        "hidden_layer_sizes" : (15,)
    },
    {
        "solver": "sgd",
        "alpha" : 1e-10,
        "activation": 'logistic',
        "random_state" : 5,
        "max_iter": 15000,
        "learning_rate_init": 0.1,
        "hidden_layer_sizes" : (20,)
    },
    {
        "solver": "sgd",
        "alpha" : 1e-10,
        "activation": 'logistic',
        "random_state" : 5,
        "max_iter": 15000,
        "learning_rate_init": 0.1,
        "hidden_layer_sizes" : (5,2)
    },
    {
        "solver": "sgd",
        "alpha" : 1e-10,
        "activation": 'logistic',
        "random_state" : 5,
        "max_iter": 15000,
        "learning_rate_init": 0.1,
        "hidden_layer_sizes" : (10,2)
    },
    {
        "solver": "sgd",
        "alpha" : 1e-10,
        "activation": 'logistic',
        "random_state" : 5,
        "max_iter": 15000,
        "learning_rate_init": 0.1,
        "hidden_layer_sizes" : (15,2)
    },
    {
        "solver": "sgd",
        "alpha" : 1e-10,
        "activation": 'logistic',
        "random_state" : 5,
        "max_iter": 15000,
        "learning_rate_init": 0.1,
        "hidden_layer_sizes" : (20,2)
    },
]

labels = [
    "1 layer, 5 neurons, logistic",
    "1 layer, 10 neurons, logistic",
    "1 layer, 15 neurons, logistic",
    "1 layer, 20 neurons, logistic",
    "2 layers, 5 neurons, logistic",
    "2 layers, 10 neurons, logistic",
    "2 layers, 15 neurons, logistic",
    "2 layers, 20 neurons, logistic",
]

plot_args = [
    {"c": "red", "linestyle": "-"},
    {"c": "green", "linestyle": "-"},
    {"c": "blue", "linestyle": "-"},
    {"c": "red", "linestyle": "--"},
    {"c": "green", "linestyle": "--"},
    {"c": "blue", "linestyle": "--"},
    {"c": "black", "linestyle": "-"},
]


def plot_on_dataset(X, y, ax, name):
    # for each dataset, plot learning for each learning strategy
    print("\nlearning on dataset %s" % name)
    ax.set_title(name)

    X = MinMaxScaler().fit_transform(X)
    mlps = []

    for label, param in zip(labels, params):
        print("training: %s" % label)
        mlp =  MLPClassifier(**param)

        # some parameter combinations will not converge as can be seen on the
        # plots so they are ignored here
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=ConvergenceWarning, module="sklearn"
            )
            mlp.fit(X, y)

        mlps.append(mlp)
        print("Training set score: %f" % mlp.score(X, y))
        print("Training set loss: %f" % mlp.loss_)
    for mlp, label, args in zip(mlps, labels, plot_args):
        ax.plot(mlp.loss_curve_, label=label, **args)

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

def get_features_song(f):
    global nb_features
    try:
        features = np.genfromtxt(f, delimiter=',')[:nb_features]
        if (len(features) == nb_features):
            return features
        return []
    except:
        return []

# function to get all the tracks from a genre in labels variable
def get_genre_songs(genre, limits=1000):
    global genre_labels
    paths = []
    tmp = genre_labels.loc[genre_labels['genre'] == genre]
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

def display_details_compute(genres, arr_nb_songs_by_genre):
    for i in range(len(genres)):
        print("{} songs in {} genre".format(arr_nb_songs_by_genre[i], genres[i]))

def generate_features_and_labels(nb_music_by_genre):
    global nb_genres
    all_features = []
    all_labels = []

    GENRES = ['Electronic', 'Experimental', 'Folk', 'Hip-Hop',
            'Instrumental', 'International', 'Pop', 'Rock']
    arr_nb_songs_by_genre = []
    start_time = time.time() # Calc time to compute
    for genre in GENRES:
        songs_computed = 0
        sound_files = get_genre_songs(genre, limits=nb_music_by_genre) # 100
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
    # print(min([np.shape(i) for i in all_features]))
    print(display_details_compute(GENRES, arr_nb_songs_by_genre))
    return np.stack(all_features), onehot_labels

features, genre_labels = generate_features_and_labels(nb_music_by_genre)

training_split = 0.8

alldata = np.column_stack((features, genre_labels))

np.random.shuffle(alldata)
splitidx = int(len(alldata) * training_split)
train, test = alldata[:splitidx,:], alldata[splitidx:,:]

train_input = train[:,:-nb_genres]
train_labels = train[:,-nb_genres:].astype(int)

test_input = test[:,:-nb_genres]
test_labels = test[:,-nb_genres:].astype(int)

nb_hidden_layer_sizes = (15,)

for ax in axes.ravel():
    plot_on_dataset(train_input, train_labels, ax=ax, name="hello")

fig.legend(ax.get_lines(), labels, ncol=3, loc="upper center")
plt.show()