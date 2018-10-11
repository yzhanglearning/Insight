## fast.ai implementation of NLP transfer learning


import os
import sys
from os.path import join as Path

from src.ingestion.data_ingestion import *
from src.preprocess.language_model import *
from src.analysis.model_pretrain import *
from src.analysis.classify import *

#################### data ingestion ########################

path = 'data/raw/Amazon'
train = DataLoad(path, 'train.ft.txt')
test = DataLoad(path, 'test.ft.txt')

print(f'The train data contains {len(train)} examples')
print(f'The test data contains {len(test)} examples')

train = train[:1000]
test = test[:1000]

trn_texts, trn_labels = DataLabelFormat(path, train, 'train')
val_texts, val_labels = DataLabelFormat(path, test, 'test')

###################### tokenization in language model ############

lm_path = path + '/amazon_lm'
tok_trn, tok_val = tokenization(trn_texts, val_texts, lm_path, tok_exist=True)

trn_lm, val_lm, itos, vs = MostFreqTok(tok_trn, tok_val, lm_path)

################# load pre-trained language model #################

pre_path = Path(path, 'models/wt103')
pre_lm_path = Path(pre_path, 'fwd_wt103.h5')

wgts, enc_wgts, row_m = preTrainModel(pre_lm_path)

em_sz, nh, nl = 400, 1150, 3

wgts = newEmbed(vs, em_sz, wgts, enc_wgts, row_m, itos, pre_path)

modelTuning(path, wgts, trn_lm, val_lm, vs, em_sz, nh, nl, emb_tune=True, sys_tune=False)
modelTuning(path, wgts, trn_lm, val_lm, vs, em_sz, nh, nl, emb_tune=True, sys_tune=True)


####################### classification #######################

class_path = Path(path, 'amazon_class')
lm_path = Path(path, 'amazon_lm')

chunk_size = 1000
preClassify(chunk_size, class_path, lm_path)

trn_size = 10
val_size = 5
classifier(trn_size, val_size, class_path, lm_path, path, itos)
