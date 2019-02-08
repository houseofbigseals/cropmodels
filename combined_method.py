import numpy as np
import growing_4 as grow
import pickle
import time
from multiprocessing import Pool
import traceback
import sys

def combined_method():
    crop = grow.plant_model()


if __name__ == "__main__":
    # dict of parameters for model from experiment
    # dt is 20 minutes and it is our new time constant
    # parameters = {'a':0.0014, 'b':0.175, 'e1':1, 'e2':10, 'k':0.15, 'dt':1}

    # make data as list of dicts
    p1 = {'a': 0.0014, 'b': 0.087, 'e1': 1, 'e2': 1, 'k': 0.15, 'dt': 1}
    p2 = {'a': 0.00055, 'b': 0.087, 'e1': 1, 'e2': 10, 'k': 0.15, 'dt': 1}
    p3 = {'a': 0.0014, 'b': 0.087, 'e1': 1, 'e2': 10, 'k': 0.15, 'dt': 1}
    p4 = {'a': 0.00055, 'b': 0.087, 'e1': 1, 'e2': 1, 'k': 0.15, 'dt': 1}
    p5 = {'a': 0.0014, 'b': 0.175, 'e1': 1, 'e2': 1, 'k': 0.15, 'dt': 1}
    p6 = {'a': 0.00055, 'b': 0.175, 'e1': 1, 'e2': 10, 'k': 0.15, 'dt': 1}
    p7 = {'a': 0.0014, 'b': 0.175, 'e1': 1, 'e2': 10, 'k': 0.15, 'dt': 1}
    p8 = {'a': 0.00055, 'b': 0.175, 'e1': 1, 'e2': 1, 'k': 0.15, 'dt': 1}

    data = [p1, p2, p3, p4, p5, p6, p7, p8]

