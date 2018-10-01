#####################  inference ######################

import pickle
import os
import sys
from os.path import join as Path

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, \
confusion_matrix

from src.ingestion.data_ingestion import *
from src.preprocess.language_model import *
from src.analysis.model_pretrain import *
from src.analysis.classify import *

sys.path.append(Path(os.getcwd(), 'tests/analysis'))
from inference import get_sentiment
from inference import prediction
from inference import experiment




####### some path #########

path = 'data/raw/Amazon'
lm_path = path + '/amazon_lm'
class_path = path + '/amazon_class'
model_path = ""



sentence = "I like Feedly"



################## user defined parameters ################

bs,bptt,em_sz,nh,nl = 52,70,400,1150,3
dps = np.array([0.4, 0.5, 0.05, 0.3, 0.1])

f = open(Path(lm_path, 'tmp/itos.pkl'), 'rb')
itos = pickle.load(f)
stoi = collections.defaultdict(lambda:0, {v:k for k,v in enumerate(itos)})

vs = len(itos)
c = int(np.load(Path(class_path, 'tmp/trn_labels.npy')).max())+1


trn_lm = np.load(Path(lm_path, 'tmp/trn_ids.npy'))
val_lm = np.load(Path(lm_path, 'tmp/val_ids.npy'))


trn_dl = LanguageModelLoader(np.concatenate(trn_lm), bs, bptt)
val_dl = LanguageModelLoader(np.concatenate(val_lm), bs, bptt)
md = LanguageModelData(path, 1, vs, trn_dl, val_dl, bs=bs, bptt=bptt)



############## model for inference ###################

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
wd = 1e-7
wd = 0

learn.load_encoder(Path(model_path, 'lm1_enc'))
learn.load(Path(model_path, 'clas_2'))








start = time.time()

print(get_sentiment(m, stoi, sentence))

print(time.time() - start)


#


trn_class = np.load(Path(class_path, 'tmp/trn_ids.npy'))
val_class = np.load(Path(class_path, 'tmp/val_ids.npy'))

trn_labels = np.squeeze(np.load(Path(class_path, 'tmp/trn_labels.npy')))
val_labels = np.squeeze(np.load(Path(class_path, 'tmp/val_labels.npy')))


# y = prediction(m, list(val_clas))
#
#
# print(f'Accuracy --> {accuracy_score(y, val_labels)}')
# print(f'Precision --> {precision_score(y, val_labels)}')
# print(f'F1 score --> {f1_score(y, val_labels)}')
# print(f'Recall score --> {recall_score(y, val_labels)}')
# print(confusion_matrix(y, val_labels))
# print(classification_report(y, val_labels))



#



from time import time
val_size = 100000
for trn_size in [50, 100, 500, 1000, 5000, 10000, 20000, 50000]:
    print('#'*50)
    print(f'Experiment with training size {trn_size}')
    start = time()
    experiment(trn_size, val_size, trn_class, trn_labels)
    t = time() - start
    print(f'Time cost: {t}')


import matplotlib.pyplot as plt

best_acc = [0.84558, 0.87324, 0.91232, 0.9203, 0.93174, 0.93584, 0.94032, 0.94616]
sizes = [50, 100, 500, 1000, 5000, 10000, 20000, 50000]
plt.plot(sizes, best_acc)
plt.title('Evolution of performance when increasing the training size')
plt.xlabel('Training size')
plt.ylabel('Accuracy')
plt.show()

plt.plot(sizes, best_acc)
plt.title('Evolution of performance when increasing the training size, Zoom on the [0-10000] size zone')
plt.xlabel('Training size')
plt.ylabel('Accuracy')
plt.xlim([0, 10000])
plt.show()

plt.plot(np.log(sizes)/np.log(10), best_acc)
plt.title('Evolution of performance when increasing the training size, with log scale for size')
plt.xlabel('Training size (log)')
plt.ylabel('Accuracy')
plt.show()







##
