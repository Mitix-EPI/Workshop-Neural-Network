#!/usr/bin/env python3

import sys
import os

command = "curl -O https://os.unil.cloud.switch.ch/fma/fma_small.zip && echo \"ade154f733639d52e35e32f5593efe5be76c6d70  fma_small.zip\"  | sha1sum -c - && unzip fma_small.zip && rm fma_small.zip"

if __name__ == '__main__':
    if (os.path.isdir("fma_small")):
        print("Music already installed.")
    else:
        os.system(command)