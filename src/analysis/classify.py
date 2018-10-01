############### going back to classification #############
from fastai.text import *
from os.path import join as Path
import numpy as np
import pandas as pd
import os
import pickle
import collections
import random

from src.preprocess.language_model import fixup
from src.preprocess.language_model import get_texts
from src.preprocess.language_model import get_all




def preClassify(class_path, lm_path):

    chunksize = 24000
    df_trn = pd.read_csv(Path(class_path, 'train.csv'), header=None, chunksize=chunksize)
    df_val = pd.read_csv(Path(class_path, 'test.csv'), header=None, chunksize=chunksize)

    tok_trn, trn_labels = get_all(df_trn, 1)
    tok_val, val_labels = get_all(df_val, 1)

    os.makedirs(Path(class_path, 'tmp'), exist_ok=True)
#(CLAS_PATH/'tmp').mkdir(exist_ok=True)

    np.save(Path(class_path, 'tmp/tok_trn.npy'), tok_trn)
    np.save(Path(class_path, 'tmp/tok_val.npy'), tok_val)

    np.save(Path(class_path, 'tmp/trn_labels.npy'), trn_labels)
    np.save(Path(class_path, 'tmp/val_labels.npy'), val_labels)

    tok_trn = np.load(Path(class_path, 'tmp/tok_trn.npy'))
    tok_val = np.load(Path(class_path, 'tmp/tok_val.npy'))
    f = open(Path(lm_path, 'tmp/itos.pkl'), 'rb')
    itos = pickle.load(f)
    stoi = collections.defaultdict(lambda:0, {v:k for k,v in enumerate(itos)})
    len(itos)


    trn_clas = np.array([[stoi[o] for o in p] for p in tok_trn])
    val_clas = np.array([[stoi[o] for o in p] for p in tok_val])

    np.save(Path(class_path, 'tmp/trn_ids.npy'), trn_clas)
    np.save(Path(class_path, 'tmp/val_ids.npy'), val_clas)



################# classifier ####################

# We select here the 'size' first reviews of our dataset
# The paper claims that it's possible to achieve very good results with few labeled examples
# So let's try with 100 examples for training, and 5000 examples for validation.
# We encourage you to try different values to see the effect of data size on performance.

def classifier(trn_size, val_size, class_path, lm_path, path, itos):

# trn_size = 100
# val_size = 50
    trn_clas = np.load(Path(class_path, 'tmp/trn_ids.npy'))
    val_clas = np.load(Path(class_path, 'tmp/val_ids.npy'))

    trn_labels = np.squeeze(np.load(Path(class_path, 'tmp/trn_labels.npy')))
    val_labels = np.squeeze(np.load(Path(class_path, 'tmp/val_labels.npy')))

    train = random.sample(list(zip(trn_clas, trn_labels)), trn_size)
    trn_clas = np.array([item[0] for item in train])
    trn_labels = np.array([item[1] for item in train])
    del train

    validation = random.sample(list(zip(val_clas, val_labels)), val_size)
    val_clas = np.array([item[0] for item in validation])
    val_labels = np.array([item[1] for item in validation])
    del validation


    bptt,em_sz,nh,nl = 70,400,1150,3
    vs = len(itos)
    opt_fn = partial(optim.Adam, betas=(0.8, 0.99))
    bs = 48

    min_lbl = trn_labels.min()
    trn_labels -= min_lbl
    val_labels -= min_lbl
    c=int(trn_labels.max())+1

    # Ccheck that the validation dataset is well balanced so acccuracy is a good metric
    # We'll also check other metrics usual for binary classification (precision, recall, f1 score)
    len(trn_labels[trn_labels == 1]) / len(trn_labels)


    trn_ds = TextDataset(trn_clas, trn_labels)
    val_ds = TextDataset(val_clas, val_labels)
    trn_samp = SortishSampler(trn_clas, key=lambda x: len(trn_clas[x]), bs=bs//2)
    val_samp = SortSampler(val_clas, key=lambda x: len(val_clas[x]))
    trn_dl = DataLoader(trn_ds, bs//2, transpose=True, num_workers=1, pad_idx=1, sampler=trn_samp)
    val_dl = DataLoader(val_ds, bs, transpose=True, num_workers=1, pad_idx=1, sampler=val_samp)



    # We define the model, here it a classifier on top of an RNN language model
    # We load the language model encoder that we fine tuned before
    # We freeze everything but the last layer, so that we can train the classification layer only.
    #load the saved weights from before, and freeze everything until the last layer

    md = ModelData(path, trn_dl, val_dl)
    dps = np.array([0.4, 0.5, 0.05, 0.3, 0.1])

    m = get_rnn_classifer(bptt, 20*70, c, vs, emb_sz=em_sz, n_hid=nh, n_layers=nl, pad_token=1, layers=[em_sz*3, 50, c], drops=[dps[4], 0.1], dropouti=dps[0], wdrop=dps[1], dropoute=dps[2], dropouth=dps[3])

    opt_fn = partial(optim.Adam, betas=(0.7, 0.99))

    learn = RNN_Learner(md, TextModel(to_gpu(m)), opt_fn=opt_fn)
    learn.reg_fn = partial(seq2seq_reg, alpha=2, beta=1)
    learn.clip=25.
    learn.metrics = [accuracy]

    lr=3e-3
    lrm = 2.6
    lrs = np.array([lr/(lrm**4), lr/(lrm**3), lr/(lrm**2), lr/lrm, lr])

    lrs=np.array([1e-4,1e-4,1e-4,1e-3,1e-2])

    wd = 1e-7
    wd = 0
    learn.load_encoder('lm1_enc')

    learn.freeze_to(-1)


    learn.lr_find(lrs/1000)



    learn.sched.plot()



    # Run one epoch on the classification layer
    learn.fit(lrs, 1, wds=wd, cycle_len=1, use_clr=(8,3))



    # Save the trained model# Save t
    learn.save('clas_0')
    learn.load('clas_0')



    # Gradually unfreeze another layer to train a bit more parameters than just the classifier layer# Gradua
    learn.freeze_to(-2)
    learn.fit(lrs, 1, wds=wd, cycle_len=1, use_clr=(8,3))



    # Save the trained model
    learn.save('clas_1')
    learn.load('clas_1')


    # Unfreeze everything and train for a few epochs on the whole set of parameters of the model
    learn.unfreeze()
    learn.fit(lrs, 1, wds=wd, cycle_len=2, use_clr=(32,10)) # cycle_len = 14


    learn.sched.plot_loss()



    # Save the model# Save t
    learn.save('clas_2')
