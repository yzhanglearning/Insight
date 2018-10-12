##################### Inference ##################

from fastai.text import *
import os, sys
import time
from os.path import join as Path

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, \
confusion_matrix


# m = get_rnn_classifer(bptt, 20*70, c, vs, emb_sz=em_sz, n_hid=nh, n_layers=nl, pad_token=1,
#           layers=[em_sz*3, 50, c], drops=[dps[4], 0.1],
#           dropouti=dps[0], wdrop=dps[1], dropoute=dps[2], dropouth=dps[3])
# opt_fn = partial(optim.Adam, betas=(0.7, 0.99))
# learn = RNN_Learner(md, TextModel(to_gpu(m)), opt_fn=opt_fn)
# learn.reg_fn = partial(seq2seq_reg, alpha=2, beta=1)
# learn.clip=25.
# learn.metrics = [accuracy]
#
# lr=3e-3
# lrm = 2.6
# lrs = np.array([lr/(lrm**4), lr/(lrm**3), lr/(lrm**2), lr/lrm, lr])
# wd = 1e-7
# wd = 0
#
# learn.load_encoder(Path(model_path, 'lm1_enc'))
# learn.load(Path(model_path, 'clas_2'))


def get_sentiment(m, stoi, input_str: str):
    # predictions are done on arrays of input.
    # We only have a single input, so turn it into a 1x1 array
    texts = [input_str]
    # tokenize using the fastai wrapper around spacy
    tok = [t.split() for t in texts]
    # tok = Tokenizer().proc_all_mp(partition_by_cores(texts))
    # turn into integers for each word
    encoded = [stoi[p] for p in tok[0]]
    idx = np.array(encoded)[None]
    idx = np.transpose(idx)
    tensorIdx = VV(idx)
    m.eval()
    m.reset()
    p = m.forward(tensorIdx)

    score = np.argmax(p[0][0].data.cpu().numpy())
    if score == 0:
        return 0 #('bad/negative')
    if score == 1:
        return 1#('good/positive')


    # return np.argmax(p[0][0].data.cpu().numpy())

def prediction(m, texts):
    """Do the prediction on a list of texts
    """
    y = []
    for i, text in enumerate(texts):
        #print(texts)
        #if i % 1000 == 0:
            #print(i)
        encoded = text
        idx = np.array(encoded)[None]
        idx = np.transpose(idx)
        #print(idx)
        tensorIdx = VV(idx)
        m.eval()
        m.reset()
        p = m.forward(tensorIdx)
        y.append(np.argmax(p[0][0].data.cpu().numpy()))
    return y


# sentence = "I like Feedly"
# start = time.time()
#
# print(get_sentiment(stoi, sentence))
#
# print(time.time() - start)
#
#
#
# y = prediction(list(val_clas))



# Show relevant metrics for binary classification
# We encourage you to try training the classifier with different data size and its effect on performance
# print(f'Accuracy --> {accuracy_score(y, val_labels)}')
# print(f'Precision --> {precision_score(y, val_labels)}')
# print(f'F1 score --> {f1_score(y, val_labels)}')
# print(f'Recall score --> {recall_score(y, val_labels)}')
# print(confusion_matrix(y, val_labels))
# print(classification_report(y, val_labels))





# trn_clas = np.load(CLAS_PATH/'tmp'/'trn_ids.npy')
# val_clas = np.load(CLAS_PATH/'tmp'/'val_ids.npy')
#
# trn_labels = np.squeeze(np.load(CLAS_PATH/'tmp'/'trn_labels.npy'))
# val_labels = np.squeeze(np.load(CLAS_PATH/'tmp'/'val_labels.npy'))


def experiment(trn_size, trn_clas, trn_labels, val_size, val_clas, val_labels, itos, trn_dl, val_dl, md):
    train = random.sample(list(zip(trn_clas, trn_labels)), trn_size)
    aux_trn_clas = np.array([item[0] for item in train])
    aux_trn_labels = np.array([item[1] for item in train])
    del train
    validation = random.sample(list(zip(val_clas, val_labels)), val_size)
    aux_val_clas = np.array([item[0] for item in validation])
    aux_val_labels = np.array([item[1] for item in validation])
    del validation
    bptt,em_sz,nh,nl = 70,400,1150,3
    vs = len(itos)
    opt_fn = partial(optim.Adam, betas=(0.8, 0.99))
    bs = 48
    min_lbl = aux_trn_labels.min()
    aux_trn_labels -= min_lbl
    aux_val_labels -= min_lbl
    c=int(aux_trn_labels.max())+1
    # Load data in relevant structures
    trn_ds = TextDataset(aux_trn_clas, aux_trn_labels)
    val_ds = TextDataset(aux_val_clas, aux_val_labels)
    trn_samp = SortishSampler(aux_trn_clas, key=lambda x: len(aux_trn_clas[x]), bs=bs//2)
    val_samp = SortSampler(aux_val_clas, key=lambda x: len(aux_val_clas[x]))
    #trn_dl = DataLoader(trn_ds, bs//2, transpose=True, num_workers=1, pad_idx=1, sampler=trn_samp)
    #val_dl = DataLoader(val_ds, bs, transpose=True, num_workers=1, pad_idx=1, sampler=val_samp)
    # Define the model and load the backbone lamguage model
    #md = ModelData(PATH, trn_dl, val_dl)
    dps = np.array([0.4, 0.5, 0.05, 0.3, 0.1])
    m = get_rnn_classifer(bptt, 20*70, c, vs, emb_sz=em_sz, n_hid=nh, n_layers=nl, pad_token=1,
              layers=[em_sz*3, 50, c], drops=[dps[4], 0.1],
              dropouti=dps[0], wdrop=dps[1], dropoute=dps[2], dropouth=dps[3])
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
    # Find th learning rate
    learn.lr_find(lrs/1000)
    # Run one epoch on the classification layer
    learn.fit(lrs, 1, wds=wd, cycle_len=1, use_clr=(8,3))
    # Save the trained model
    learn.save(f'{trn_size}clas_0')
    learn.load(f'{trn_size}clas_0')
    # Gradually unfreeze another layer to train a bit more parameters than just the classifier layer
    learn.freeze_to(-2)
    learn.fit(lrs, 1, wds=wd, cycle_len=1, use_clr=(8,3))
    # Save the trained model
    learn.save(f'{trn_size}clas_1')
    learn.load(f'{trn_size}clas_1')
    # Unfreeze everything and train for a few epochs on the whole set of parameters of the model
    learn.unfreeze()
    learn.fit(lrs, 1, wds=wd, cycle_len=14, use_clr=(32,10))
    # Save the model
    learn.sched.plot_loss()
    learn.save(f'{trn_size}clas_2')
