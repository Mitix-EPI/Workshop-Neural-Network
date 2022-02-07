"""Adapted from https://github.com/mdeff/fma/blob/master/utils.py"""

import os
import ast

import pandas as pd

def get_audio_path(track_id, solution=False):
    tid_str = '{:06d}'.format(track_id)
    if (solution == False):
        return os.path.join('data', 'fma_small', tid_str[:3], tid_str + '.csv')
    else:
        return os.path.join('../data', 'fma_small', tid_str[:3], tid_str + '.csv')
