
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


#

def load_file(filename):
    with open(filename, 'r', encoding="utf8", errors="ignore") as f:  # , encoding='utf-8'
        data = f.readlines()
    return data

def encode_data(data, tokenizer, punctuation_enc):
    """
    Converts words to (BERT) tokens and puntuation to given encoding.
    Note that words can be composed of multiple tokens.

    there	O
    is	O
    very	O
    little	O
    20th	O
    century	O
    technology	O
    in	O
    them	PERIOD


    ->

    2045	0
    2003	0
    2200	0
    2210	0
    3983	0
    2301	0
    2974	0
    1999	0
    2068	2

    """
    X = []
    Y = []
    for line in data:
        word, punc = line.split('\t')
        punc = punc.strip()
        tokens = tokenizer.tokenize(word)
        x = tokenizer.convert_tokens_to_ids(tokens)
        y = [punctuation_enc[punc]]
        if len(x) > 0:
            if len(x) > 1:
                y = (len(x)-1)*[0]+y
            X += x
            Y += y
    return X, Y

def insert_target(x, segment_size):
    """
    Creates segments of surrounding words for each word in x.
    Inserts a zero token halfway the segment.

    Let segment is 8.

    2045	0
    2003	0
    2200	0
    2210	0
    3983	0
    2301	0
    2974	0
    1999	0
    2068	2

    ->

    [1999, 2068, 2045,    0, 2003, 2200, 2210, 3983]    0
    [2068, 2045, 2003,    0, 2200, 2210, 3983, 2301]    0
    [2045, 2003, 2200,    0, 2210, 3983, 2301, 2974]    0
    [2003, 2200, 2210,    0, 3983, 2301, 2974, 1999]    0
    [2200, 2210, 3983,    0, 2301, 2974, 1999, 2068]    0
    [2210, 3983, 2301,    0, 2974, 1999, 2068, 2045]    0
    [3983, 2301, 2974,    0, 1999, 2068, 2045, 2003]    0
    [2301, 2974, 1999,    0, 2068, 2045, 2003, 2200]    0
    [2974, 1999, 2068,    0, 2045, 2003, 2200, 2210]    2


The logic behind is the following. BERT cannot work with sentences with different lengths. But we could do padding for each sentence. But we can not do it because we don't have sentences. We just have text without punctuation. So we have to take a bunch of words and do training on them. We could take the first words and the next ten. But how in such a situation can we deal with words at the end of the text?  It is why we send 16 words from end to beginning and vice versa. Next. Important thing is to understand how dealing with prediction. We use one padding to show to BERT the most important place in each pattern.
        """

    ########################################################################
    # original
    # X = []
    # x_pad = x[-((segment_size-1)//2-1):]+x+x[:segment_size//2]
    # for i in range(len(x_pad)-segment_size+2):
    #     segment = x_pad[i:i+segment_size-1]
    #     segment.insert((segment_size-1)//2, 0)
    #     X.append(segment)
    #######################################################################

    #######################################################################
    # without padding
    # X = []
    # x_pad = x[-(((segment_size+1)-1)//2-1):]+x+x[:(segment_size+1)//2]
    # for i in range(len(x_pad)-(segment_size+1)+2):
    #     segment = x_pad[i:i+segment_size+1-1]
    #     # segment.insert((segment_size-1)//2, 0)
    #     X.append(segment)
    ########################################################################

    print('my method')
    ########################################################################
    # without padding and mirror change for fist and last 16 tokens
    x_segment = []
    for i in range(len(x)):
        if i >= segment_size//2 and i <= len(x) - segment_size//2:
            x_segment.append(x[i-segment_size//2:i+segment_size//2])
        if i < segment_size//2:
            x_segment.append(x[:segment_size])
        if i > len(x) - segment_size//2:
            x_segment.append(x[len(x)-segment_size:])
    ########################################################################


    return np.array(x_segment)

def preprocess_data(data, tokenizer, punctuation_enc, segment_size):
    X, y = encode_data(data, tokenizer, punctuation_enc)
    X = insert_target(X, segment_size)
    return X, y

def create_data_loader(X, y, shuffle, batch_size):
    data_set = TensorDataset(torch.from_numpy(X).long(), torch.from_numpy(np.array(y)).long())
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=shuffle)
    return data_loader
