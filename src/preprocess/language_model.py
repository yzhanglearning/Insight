################ language model ###############
"""
freqs: A collections.Counter object holding the frequencies of tokens
       in the data used to build the Vocab.
stoi: A collections.defaultdict instance mapping token strings to
       numerical identifiers.
itos: A list of token strings indexed by their numerical identifiers.

"""


# We're going to fine tune the language model so it's ok to take some of the test set in our train data
# for the lm fine-tuning
from fastai.text import *
from sklearn.model_selection import train_test_split
import numpy as np
from os.path import join as Path
import re
import pandas as pd
import html



BOS = 'xbos' # beginning-of-sentence tag
FLD = 'xfld' # data field SyntaxWarning
col_names = ['labels', 'texts']


# Here we use functions from the fast.ai course to get data
chunksize = 24000
re1 = re.compile(r' +')

def fixup(x):
    x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
        'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
        '<br />', "\n").replace('\\"', '"').replace('<unk>','u_n').replace(' @.@ ','.').replace(
        ' @-@ ','-').replace('\\', ' \\ ')
    return re1.sub(' ', html.unescape(x))

def get_texts(df, n_lbls=1):
    labels = df.iloc[:,range(n_lbls)].values.astype(np.int64)
    texts = f'\n{BOS} {FLD} 1 ' + df[n_lbls].astype(str)
    for i in range(n_lbls+1, len(df.columns)):
        texts += f' {FLD} {i-n_lbls} ' + df[i].astype(str)
    texts = list(texts.apply(fixup).values)
    tok = Tokenizer().proc_all_mp(partition_by_cores(texts))
    return tok, list(labels)

def get_all(df, n_lbls):
    tok, labels = [], []
    for i, r in enumerate(df):
        print(i)
        tok_, labels_ = get_texts(r, n_lbls)
        tok += tok_;
        labels += labels_
    return tok, labels



# function of tokenization
def tokenization(trn_texts, val_texts, lm_path, tok_exist=True):



    trn_texts, val_texts = train_test_split(np.concatenate([trn_texts, val_texts]), test_size=0.1)

    df_trn = pd.DataFrame({'text': trn_texts, 'labels': [0]*len(trn_texts)}, columns=col_names)
    df_val = pd.DataFrame({'text': val_texts, 'labels': [0]*len(val_texts)}, columns=col_names)

    df_trn.to_csv(Path(lm_path, 'train.csv'), header=False, index=False)
    df_val.to_csv(Path(lm_path, 'test.csv'), header=False, index=False)







    df_trn = pd.read_csv(Path(lm_path, 'train.csv'), header=None, chunksize=chunksize)
    df_val = pd.read_csv(Path(lm_path, 'test.csv'), header=None, chunksize=chunksize)

    # This cell can take quite some time if your dataset is large
    # Run it once and comment it for later use
    tok_trn, trn_labels = get_all(df_trn, 1)
    tok_val, val_labels = get_all(df_val, 1)


    # Run this cell once and comment everything but the load statements for later use

    if tok_exist:
        tok_trn = np.load(Path(lm_path, 'tmp/tok_trn.npy'))
        tok_val = np.load(Path(lm_path, 'tmp/tok_val.npy'))
    else:
        os.makedirs(Path(lm_model, 'tmp'), exist_ok=True)
        np.save(Path(lm_model, 'tmp/tok_trn.npy'))
        np.save(Path(lm_model, '/tmp/tok_val.npy'))

    return tok_trn, tok_val


# function to check the most frequent tokens
def MostFreqTok(tok_trn, tok_val, lm_path):

    # check the most frequent tokens
    freq = Counter(p for o in tok_trn for p in o)
    freq.most_common(25)

    # check the least common tokens
    freq.most_common()[-25:]

    # Build your vocabulary by keeping only the most common tokens that appears frequently enough
    # and constrain the size of your vocabulary. We follow here the 60k recommendation.
    max_vocab = 60000
    min_freq = 2

    itos = [o for o,c in freq.most_common(max_vocab) if c > min_freq]
    itos.insert(0, '_pad_')
    itos.insert(0, '_unk_')

    stoi = collections.defaultdict(lambda: 0, {v:k for k, v in enumerate(itos)})
    len(itos)

    trn_lm = np.array([[stoi[o] for o in p] for p in tok_trn])
    val_lm = np.array([[stoi[0] for o in p] for p in tok_val])

    np.save(lm_path, 'tmp/trn_ids.npy', trn_lm)
    np.save(lm_path, 'tmp/val_idx.npy', val_lm)
    pickle.dump(itos, open(Path(lm_path, 'tmp/itos.pkl'), 'wb'))

    # save everything
    trn_lm = np.load(Path(lm_path, 'tmp/trn_ids.npy'))
    val_lm = np.load(Path(lm_path, 'tmp/val_ids.npy'))
    itos = pickle.load(open(Path(lm_path, 'tmp/itos.pkl'), 'rb'))

    vs = len(itos)
    print(vs, len(trn_lm))

    return trn_lm, val_lm, itos, vs
