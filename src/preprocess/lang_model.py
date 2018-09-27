
################ language model ###############
import numpy as np
from sklearn.model_selection import train_test_split

import sys
import os
import re
import html
from fastai.text import *


cwd = os.getcwd()
sys.path.append(cwd+'/src/ingestion')
from data_ingestion import *


# We're going to fine tune the language model so it's ok to take some of the test set in our train data
# for the lm fine-tuning
trn_texts,val_texts = train_test_split(np.concatenate([trn_texts,val_texts]), test_size=0.1)

df_trn = pd.DataFrame({'text':trn_texts, 'labels':[0]*len(trn_texts)}, columns=col_names)
df_val = pd.DataFrame({'text':val_texts, 'labels':[0]*len(val_texts)}, columns=col_names)

df_trn.to_csv(os.path.join(LM_PATH, 'train.csv'), header=False, index=False)
df_val.to_csv(os.path.join(LM_PATH, 'test.csv'), header=False, index=False)


# Here we use functions from the fast.ai course to get data

chunksize=24000
re1 = re.compile(r'  +')

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

df_trn = pd.read_csv(os.path.join(LM_PATH, 'train.csv'), header=None, chunksize=chunksize)
df_val = pd.read_csv(os.path.join(LM_PATH, 'test.csv'), header=None, chunksize=chunksize)



# This cell can take quite some time if your dataset is large
# Run it once and comment it for later use
tok_trn, trn_labels = get_all(df_trn, 1)
tok_val, val_labels = get_all(df_val, 1)


# Run this cell once and comment everything but the load statements for later use


#(LM_PATH/'tmp').mkdir(exist_ok=True)
#np.save(LM_PATH/'tmp'/'tok_trn.npy', tok_trn)
#np.save(LM_PATH/'tmp'/'tok_val.npy', tok_val)
tok_trn = np.load(os.path.join(LM_PATH, 'tmp/'+'tok_trn.npy'))
tok_val = np.load(os.path.join(LM_PATH, 'tmp/'+'tok_val.npy'))


# Check the most common tokens
freq = Counter(p for o in tok_trn for p in o)
freq.most_common(25)

# Check the least common tokens
freq.most_common()[-25:]


# Build your vocabulary by keeping only the most common tokens that appears frequently enough
# and constrain the size of your vocabulary. We follow here the 60k recommendation.
max_vocab = 60000
min_freq = 2

itos = [o for o,c in freq.most_common(max_vocab) if c>min_freq]
itos.insert(0, '_pad_')
itos.insert(0, '_unk_')

stoi = collections.defaultdict(lambda:0, {v:k for k,v in enumerate(itos)})
len(itos)

trn_lm = np.array([[stoi[o] for o in p] for p in tok_trn])
val_lm = np.array([[stoi[o] for o in p] for p in tok_val])

np.save(os.path.join(LM_PATH, 'tmp/'+'trn_ids.npy'), trn_lm)
np.save(os.path.join(LM_PATH, 'tmp/'+'val_ids.npy'), val_lm)
pickle.dump(itos, open(os.path.join(LM_PATH, 'tmp/'+'itos.pkl'), 'wb'))


# Save everything
trn_lm = np.load(os.path.join(LM_PATH, 'tmp/'+'trn_ids.npy'))
val_lm = np.load(os.path.join(LM_PATH, 'tmp/'+'val_ids.npy'))
itos = pickle.load(open(os.path.join(LM_PATH, 'tmp/'+'itos.pkl'), 'rb'))


vs=len(itos)
vs,len(trn_lm)
