#!/usr/bin/env python3
"""
Predict if a tweet is mean or not &
the probability of it being mean.

Note: Once you've decied that your model is good enough,
the common practice is to train it with the whole data set
that you have and then use it for predictions.
"""
import os
import pickle
import sys

from trainModel import get_params, confs
from prepData import clean_data

from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


if __name__ == '__main__':
    name, epochs, batches = get_params()
    model = confs[name]
    mname = 'models/model-%s-%d-%d' % (name, epochs, batches)
    model_file = mname + '.h5'
    tokenizer_file = mname + '-tokenizer.pickle'

    # Loading the model.
    if os.path.exists(model_file):
        model = load_model(model_file)
        print("\n***** Model loaded! *****")
    else:
        print("\nCan't find %s model, train it first using 'trainModel.py %s %d %d'" % (mname, name, epochs, batches))
    
    # Loading tokenizer.
    # We need to use the same tokenizer that we've used
    # for training and testing to get the same encoding
    # for known words in our vocabulary and also, in
    # the word embedding that we've created during the training.
    if os.path.exists(tokenizer_file):
        tokenizer = pickle.load(open(mname+'-tokenizer.pickle', "rb" ))
        print('\n***** Tokenizer loaded! *****')
    else:
        print("\nCan't find tokenizer for %s model, train it first using 'trainModel.py %s %d %d'" % (mname, name, epochs, batches))
    
    # Get the tweet from the Command Line Prompt.
    print("\n----------------------------------------------------------")
    print("Type in one tweet per line and press 'Ctrl-D' (on *nix) or\n'Ctrl-Z' and then 'Enter' (on Windows) when you're done:")
    print("----------------------------------------------------------")
    for tweet in sys.stdin.readlines():
        # Cleanup the tweet before we use our model.
        t = clean_data([tweet], True)
        # Encode and pad our tweet with the same tokenizer
        # that we've used for training and testing.
        # We've set our own variable in 'tokenizer._max_padding_len'
        # on training to store info about the max length of our encoded text.
        t = tokenizer.texts_to_sequences(t)
        t = pad_sequences(t, maxlen=tokenizer._max_padding_len, padding='post')
        # Classify our tweet mood:
        # '0' - negative and '1' - positive.
        pred_class = model.predict_classes(t)
        #print("\nprob class: ", pred_class, "value: ", pred_class[0][0])
        pred_class = pred_class[0][0]
        # We can also get the probablity of prediction being in a given class.
        # By default we get the probablity of being positive. We can get the 
        # probablity of tweet being "mean" just by calculating (1-prob).
        prob = model.predict_proba(t)
        #print("\nprob.shape: ", prob.shape, "value: ", prob)
        prob = prob[0][0]

        if pred_class == 1:
            txtMood = "positive"
            probPercent = prob*100
        else:   #this is a mean tweet
            txtMood = "negative"
            probPercent = (1-prob)*100
         
        print("<%s> -> %s mood (%.2f%% probability)" % (tweet.strip(), txtMood, probPercent) )