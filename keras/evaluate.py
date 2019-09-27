import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))

import numpy as np
import time
import logging
import matplotlib.pyplot as plt
from sklearn import metrics
import _pickle as cPickle
import datetime
from utilities import d_prime

from keras_utils import forward


class Evaluator(object):
    def __init__(self, model, generator):
        self.model = model
        self.generator = generator
        
    def evaluate(self):

        # Forward
        output_dict = forward(
            model=self.model, 
            generator=self.generator, 
            return_target=True)

        outputs = output_dict['output']    # (audios_num, classes_num)
        targets = output_dict['target']    # (audios_num, classes_num)

        average_precision = metrics.average_precision_score(
            targets, outputs, average=None)

        auc = metrics.roc_auc_score(targets, outputs, average=None)
        
        statistics = {'average_precision': average_precision, 'auc': auc}
        
        return statistics
