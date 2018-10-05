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

# lr=3e-3
# lrm = 2.6
# lrs = np.array([lr/(lrm**4), lr/(lrm**3), lr/(lrm**2), lr/lrm, lr])
# wd = 1e-7
# wd = 0

learn.load_encoder(Path(model_path, 'lm1_enc'))
learn.load(Path(model_path, 'clas_2'))





# sentence = "I like this book"
# sentence = "I don't like movie"
# sentence = 'I think this is the worse CD I have ever bought, the musician is an amateru.'
# sentence = 'I think this is the perfect CD I have ever bought, the musician is a professional.'


# sentence = 'I have awful heartburn.'
# sentence = "Frightened by what I'm about to do tonight."
# sentence = "I’m feeling like my nap made me even more tired."
# sentence = "Switching between good and bad a lot lately. Problems mainly with bad cognitions."
# sentence = "I feel a jolt of anger"
# sentence = "Yes, tired of many things! Taking medicine, being on disability, never seeming to improve!"
# sentence = "I’ve been up since 5:30. I am kinda tired. I’ll probably go to bed in an hour or so."
# sentence = 'I am actually just physically tired. Long week. Not a morning person. Not enough coffee in the world to make me one. '
# sentence = "I'm also often tired of people. That's when I curl up with a good book and recharge."
# sentence = "I’ve been getting exhausted after 3 and falling deeply asleep at 7 and sleeping 10 hours straight. Considering I’m not tired before 3 and I’m getting things done during the day I don’t think it’s too much of a problem."
# sentence = "I am grateful for my hubby and kids and the fact that my daughter is learning to drive."
# sentence = "I'm grateful for my noise-cancelling headphones. The cord, not so much."
# sentence = "I'm grateful to be grabbing a bite to eat with a friend, and considering volunteering in a more responsible and challenging arena."
# sentence = "I'm grateful for a day to rest."
# sentence = "Grateful that my son came over for a visit."
# sentence = "I love the kids, don't get me wrong--but I'm grateful for them being gone this weekend."
# sentence = "well I got subscribed to a new boxing channel which shows 24 hour matches and previews. it's amazing. I always liked boxing."
# sentence = "I'm grateful for my family, even how nuts they are."
# sentence = "I've never felt like a real human, since my early childhood I said my soul is alien, from another planet. "
# sentence = "This feeling returns every now and then, I even searched for a Starseed community online but I don't really like it to be honest. "
# sentence = "I was scared on Saturday that I may really be an alien soul"
# sentence = " it seems to me I'm only faking being human"
# sentence = "But it feels fake to me. This is not my real identity. "
# sentence = "I don't sleep like a regular human"
# sentence = "I eat more like a machine than a human"
# sentence = "I don't connect/ get along with other humans (but I do with doors and walls.)"
# sentence = "I don't connect/ get along with other humans (but I do with doors and walls.)"
# sentence = "my looks resemble the older sister of shrek (who isn't human, an ogre)"
# sentence = "- I can do an impression of mickey mouse/ minie mouse/ Donald duck/ what ever Disney character, but no actual celebrities or human impressions"
# sentence = "and I technickly live in a dark cave (my room), only coming out to hunt for food/ water when it's quiet"
# sentence = "Because of my trauma/abuse I've often felt sub-human."
# sentence = "I feel human. I haven’t always felt female though."
# sentence = "Finally after leaving my bad marriage & learning why my family growing up was so dysfunctional. Have to admit that all the wonderful people now in my new environment really helped me learn what really being a caring human is all about"
# sentence = "Nowadays I feel outrageously human."
# sentence = "What makes me feel human? This glorious process of dying! I’m bedridden, now. When my brain lacks oxygen I feel high. When I’m pain-medicated I feel high. Most of the day I’m more idiotic than not."
# sentence = "yes, in our kindergarten graduation we sang a song about do you ever feel like a lunatic? a lunatic a lunatic refrain"
# sentence = "I’ve never been able to refrain from feeling like a lunatic."
sentence = "The one that quickly comes to mind is “please be kind, rewind.”"







#
# start = time.time()
# print('The analysis result: ', get_sentiment(m, stoi, sentence))
# print('Inferece time is: ', time.time() - start, 'seconds')
#
#




#







trn_class = np.load(Path(class_path, 'tmp/trn_ids.npy'))
val_class = np.load(Path(class_path, 'tmp/val_ids.npy'))

trn_labels = np.squeeze(np.load(Path(class_path, 'tmp/trn_labels.npy')))
val_labels = np.squeeze(np.load(Path(class_path, 'tmp/val_labels.npy')))



d = 5000
#print(val_class.shape)
val_class = trn_class[:d]
val_labels = trn_labels[:d]


y = prediction(m, list(val_class))
#y = prediction(m, sentence)

start = time.time()
print(y)
print(time.time() - start)



#

print(f'Accuracy --> {accuracy_score(y, val_labels)}')
print(f'Precision --> {precision_score(y, val_labels)}')
print(f'F1 score --> {f1_score(y, val_labels)}')
print(f'Recall score --> {recall_score(y, val_labels)}')
print(confusion_matrix(y, val_labels))
print(classification_report(y, val_labels))



#



# from time import time
# val_size = 1000
# for trn_size in [50, 100, 500, 1000, 5000, 10000, 20000, 50000]:
#     print('#'*50)
#     print(f'Experiment with training size {trn_size}')
#     start = time()
#     experiment(trn_size, trn_class, trn_labels, val_size, val_class, val_labels, itos, trn_dl, val_dl, md)
#     t = time() - start
#     print(f'Time cost: {t}')
#
#
# import matplotlib.pyplot as plt
#
# best_acc = [0.84558, 0.87324, 0.91232, 0.9203, 0.93174, 0.93584, 0.94032, 0.94616]
# sizes = [50, 100, 500, 1000, 5000, 10000, 20000, 50000]
# plt.plot(sizes, best_acc)
# plt.title('Evolution of performance when increasing the training size')
# plt.xlabel('Training size')
# plt.ylabel('Accuracy')
# plt.show()
#
# plt.plot(sizes, best_acc)
# plt.title('Evolution of performance when increasing the training size, Zoom on the [0-10000] size zone')
# plt.xlabel('Training size')
# plt.ylabel('Accuracy')
# plt.xlim([0, 10000])
# plt.show()
#
# plt.plot(np.log(sizes)/np.log(10), best_acc)
# plt.title('Evolution of performance when increasing the training size, with log scale for size')
# plt.xlabel('Training size (log)')
# plt.ylabel('Accuracy')
# plt.show()
#
#





##
