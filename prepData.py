#!/usr/bin/env python3
"""
Preparing data for sentiment analysis.

IMDB movie review dataset: http://www.cs.cornell.edu/people/pabo/movie-review-data
polarity dataset v2.0
"""

from os import listdir
from os import path

from stopwords import stopwords as exclude

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from pprint import pprint


def encode_text(text2encode, tokenizer, max_len=None, for_training=False, ):
    """
    Return text encoded in a way that we can use it in a neural network.
    Function parameters:
    'text2encode' - plain text to be encoded
    'tokenizer' - Python's tokenizer object
    'max_len' - the maximum length of sentence (in words), if a sentence
                is shorter than max_len pad it to match the max length

    'for_training' - determine if we're working with training data or not,
                     we do two unique things in training: fit tokenizer on 
                     the text and compute max_len value.

    An example of encoded sentence:
    Input sentence: "isn't it the ultimate sign of a movie's cinematic "
    Output encoding: [167, 9, 1, 1820, 1560, 4, 2, 603, 1064]

    isn't = 167
    it = 9
    the = 1
    ultimate = 1820
    sign = 1560
    of = 4
    a = 2
    movie's = 603
    cinematic = 1064

    Remember to use the same tokenizer to encode both train and test data,
    but you need to "fit"/prepare tokenizer before all that only on training data.
    """

    print("Tokenizing the text sequence...")
    # We "fit" our tokenizer on our training set.
    # This is where unique numbers are generated for each word.
    if for_training:
        tokenizer.fit_on_texts(text2encode)

    # Encode words(tokens) as unique numbers.
    encoded_txt = tokenizer.texts_to_sequences(text2encode)

    # We're looking for the longest sentence in our training set.
    # Then we will use it when we run encode_text() on test data.
    # The key here is to have maximum lenght all the same troughout
    # training and data sets.
    if not max_len:
        max_len = max([len(s.split()) for s in text2encode])
        tokenizer._max_padding_len = max_len

    # We need to pad our encoded text to the maximum lenght
    # for our embedding layer to work properly.
    train_x = pad_sequences(encoded_txt, maxlen=max_len, padding='post')

    if for_training:
        return train_x, max_len

    return train_x


def cleanup(word, clean_sw=True):
    """
    Return a word if it's significant and 'None' if it can be filtered out.

    'clean_sw' - should we filter out stopwords?
    """

    word = word.strip().lower()
    if not word.isalpha():
        return None
    if clean_sw and word in exclude:
        return None
    if len(word) == 1:
        return None

    return word


def clean_data(data, clean_sw):
    """
    Remove unnecessary words and characters from a data set.

    data - a list of sentences to clean
    clean_sw - should we filter out stop words?
    """
    out = []
    for doc in data:
        wout = []
        for w in doc.split():
            w = cleanup(w, clean_sw)
            if w == None:
                continue
            wout.append(w)
        out.append(' '.join(wout))

    return out


def get_data(dirr='data/txt_sentoken', do_cleanup=True, filter_stopwords=True):
    """
    Load all our data into memory, split into training and test sets,
    clean up and encode, so we can use it with our neural network.

    'do_cleanup' - should we remove insignificant characters & words?
    'filter_stopwords' - should we remove commonly used words?
    """
    train_x=[]
    train_y=[]

    test_x=[]
    test_y=[]

    # First, load all of the data into 'train_x' set.
    print('Loading data...')
    for p in ['negative', 'positive']:
        for filename in listdir(path.join(dirr, p)):
            dfile = path.join(dirr, p, filename)
            data = open(dfile).read()
            train_x.append(data)

    if do_cleanup:
        print("Cleaning up the data...")
        ct = clean_data(train_x, filter_stopwords)
    else:
        ct = train_x

    # Split our data set into training and test sets.
    # Train-to-Test sets ratio is 90% & 10% respectively,
    # so we have 900 positive and 100 negative reviews.
    l = 1000
    train_len = int(l * 0.90)
    test_len = int(l * 0.10)

    # negative reviews were added to train_x set first.
    train_x_neg = ct[0 : train_len]
    train_x_pos = ct[l : l+train_len]

    # Generate approriate labels for negative data.
    # '0' means negative, '1' - positive.
    train_y_neg = [ 0 for i in range(len(train_x_neg))]
    train_y_pos = [ 1 for i in range(len(train_x_pos))]

    # Put all of training set splits together.
    train_x = train_x_neg + train_x_pos
    train_y = train_y_neg + train_y_pos

    # Get the remining 10% of data as test set.
    test_x_neg = ct[train_len : l]
    test_x_pos = ct[l+train_len : ]

    test_y_neg = [ 0 for i in range(len(test_x_neg))]
    test_y_pos = [ 1 for i in range(len(test_x_neg))]

    test_x = test_x_neg + test_x_pos
    test_y = test_y_neg + test_y_pos

    # Create a new tokenizer obj, we will use it
    # for both - training and test data sets.
    tokenizer = Tokenizer()
    # Encode and pad our train and test data.
    input_train_x = train_x
    train_x, max_len = encode_text(train_x, tokenizer, for_training=True)
    test_x = encode_text(test_x, tokenizer, max_len=max_len)

    # Just show a sample of input text and encoded text.
    print("\nOutput from tokenizer:")
    pprint(input_train_x[0][:50])
    pprint(train_x[0][:9])

    for w in input_train_x[0][:50].replace(':','').split():
        if w in tokenizer.word_index.keys():
            print(w, '=', tokenizer.word_index[w])
    print()

    # Get a vocabulary size (a number of unique words).
    # We will later have to use it for our Embedding layer.
    inputs = len(tokenizer.word_index) + 1
    print("Vocabulary size: ", inputs)

    return train_x, train_y, test_x, test_y, inputs, max_len, tokenizer


if __name__ == '__main__':
    train_x, train_y, test_x, test_y, inputs, max_len, t = get_data()

    print('\nX[0]:\n', train_x[0])
    print('Y[0]:\n', train_y[0])
