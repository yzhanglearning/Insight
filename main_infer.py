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

####### path to data and model #########

path = 'data/raw/Mental'
lm_path = path + '/amazon_lm'
class_path = path + '/amazon_class'
model_path = ""

################## hyper-parameters ################
bs, bptt, em_sz, nh, nl = 52,70,400,1150,3
dps = np.array([0.4, 0.5, 0.05, 0.3, 0.1])

class hyperParams(object):
    def __init__(self, bs, bptt, em_sz, nh, nl, dps):
        self.bs = bs
        self.bptt = bptt
        self.em_sz = em_sz
        self.nh = nh
        self.nl = nl
        self.dps = dps

    def showParams(self):
        print('Showing user input hyper-parameters:\n')
        print('bs is: {}'.format(self.bs))
        print('bptt is: {}'.format(self.bptt))
        print('em_sz is: {}'.format(self.bptt))
        print('nh is: {}'.format(self.nh))
        print('nl is: {}'.format(self.nl))
        print('dps are: {}'.format(self.dps))


InputParams = hyperParams(bs, bptt, em_sz, nh, nl, dps)

InputParams.showParams()

################## loaded parameters ######################

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

learn.load_encoder(Path(model_path, 'lm1_enc'))
learn.load(Path(model_path, 'clas_2'))

####### read mental sentences #############

path = 'data/raw/Mental'
train = DataLoad(path, 'train.ft.txt')
test = DataLoad(path, 'test.ft.txt')

train = train[:50]
test = test[:50]

print(f'The train data contains {len(train)} examples')
print(f'The test data contains {len(test)} examples')

trn_texts, trn_labels = DataLabelFormat(path, train, 'train')
val_texts, val_labels = DataLabelFormat(path, test, 'test')

val_labels =[1 for i in range(len(train))] + [0 for i in range(len(test))]
val_class = trn_texts + val_texts
print(trn_texts)
print(val_labels)
print('total labels:', len(val_labels))
print('total classes:', len(val_class))

y = []
for i in range(len(val_labels)):
    y.append(get_sentiment(m, stoi, str(val_class[0][:-2])))
print(y)

## metrics
print(f'Accuracy --> {accuracy_score(y, val_labels)}')
print(f'Precision --> {precision_score(y, val_labels)}')
print(f'F1 score --> {f1_score(y, val_labels)}')
print(f'Recall score --> {recall_score(y, val_labels)}')
print(confusion_matrix(y, val_labels))
print(classification_report(y, val_labels))
