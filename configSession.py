"""
Configure the session to get the same results every time.

Based on:
https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
"""

import os
import logging

import numpy as np
import random as rn

import tensorflow as tf
from keras import backend as K

# (In Windows 10): To disable KMP related logging messages
# set the environment variable 'KMP_AFFINITY=disabled' and
# restart your Command Line Prompt.

# Silence Tensorflow debug & log messages
# for the clarity of the output.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').disabled = True

# Set up random seed to get the same results
# every time when training your model.
os.environ['PYTHONHASHSEED'] = '0'

rs = 5
np.random.seed(rs)
rn.seed(rs)
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
tf.compat.v1.set_random_seed(rs)

sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
K.set_session(sess)

print("\n***** Session configured! *****")
