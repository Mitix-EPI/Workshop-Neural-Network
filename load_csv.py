import os
import numpy as np

def read_directory(directory):
    files = []
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            files.append(directory + filename)
        if (os.path.isdir(directory + filename)):
            files += read_directory(directory + filename + "/")
    return files

for f in read_directory("computed/") :
    print(np.genfromtxt(f, delimiter=','))