#!/usr/bin/env python3

import os

creation_of_folder = "mkdir -f data && cd data"
install_metadata = "curl -O https://os.unil.cloud.switch.ch/fma/fma_metadata.zip && echo 'f0df49ffe5f2a6008d7dc83c6915b31835dfe733  fma_metadata.zip' | sha1sum -c - && echo 'You need to extract fma_metadata.zip manually'"
install_data = "curl -O https://os.unil.cloud.switch.ch/fma/fma_small.zip && echo 'ade154f733639d52e35e32f5593efe5be76c6d70  fma_small.zip'  | sha1sum -c - && echo 'You need to extract fma_small.zip manually'"
leave_folder = "cd .."

if __name__ == '__main__':
    if (os.path.isdir("fma_metadata") and os.path.isdir("fma_small")):
        print("Music already installed !")
    else:
        command = creation_of_folder + " && " + install_metadata + " && " + install_data + " && " + leave_folder
        os.system(command)
        print("Music installed !")