


############## using pretrained language model ##############


# Uncomment this cell to download the pre-trained model.
# It will be placed into the PATH that you defined earlier.
#! wget -nH -r -np -P {PATH} http://files.fast.ai/models/wt103/
import os, sys
import pickle

cwd = os.getcwd()
sys.path.append(cwd+'/src/preprocess')
from lang_model import *

# Load the weights of the modelI
em_sz,nh,nl = 400,1150,3

PATH = cwd + '/data/raw/Amazon/'
PRE_PATH = os.path.join(PATH, 'models/wt103/')
PRE_LM_PATH = os.path.join(PRE_PATH, 'fwd_wt103.h5')

wgts = torch.load(PRE_LM_PATH, map_location=lambda storage, loc: storage)

# Check the word embedding layer and keep a 'mean word' for unknown tokens
enc_wgts = to_np(wgts['0.encoder.weight'])
row_m = enc_wgts.mean(0)

enc_wgts.shape


print(PRE_PATH)
# Load the vocabulary on which the pre-trained model was trained
# Define an embedding matrix with the vocabulary of our dataset
f = open(os.path.join(PRE_PATH, 'itos_wt103.pkl'), 'rb')
itos2 = pickle.load(f)
#itos2 = pickle.load(os.path.join(PRE_PATH, 'itos_wt103.pkl').open('rb'))
stoi2 = collections.defaultdict(lambda:-1, {v:k for k,v in enumerate(itos2)})

new_w = np.zeros((vs, em_sz), dtype=np.float32)
for i,w in enumerate(itos):
    r = stoi2[w]
    new_w[i] = enc_wgts[r] if r>=0 else row_m


# Use the new embedding matrix for the pre-trained model
wgts['0.encoder.weight'] = T(new_w)
wgts['0.encoder_with_dropout.embed.weight'] = T(np.copy(new_w))
wgts['1.decoder.weight'] = T(np.copy(new_w))


# Define the learner object to do the fine-tuning
# Here we will freeze everything except the embedding layer, so that we can have a better
# embedding for unknown words than just the mean embedding on which we initialise it.
wd=1e-7
bptt=70
bs=52
opt_fn = partial(optim.Adam, betas=(0.8, 0.99))

trn_dl = LanguageModelLoader(np.concatenate(trn_lm), bs, bptt)
val_dl = LanguageModelLoader(np.concatenate(val_lm), bs, bptt)
md = LanguageModelData(PATH, 1, vs, trn_dl, val_dl, bs=bs, bptt=bptt)

drops = np.array([0.25, 0.1, 0.2, 0.02, 0.15])*0.7

learner= md.get_model(opt_fn, em_sz, nh, nl,
    dropouti=drops[0], dropout=drops[1], wdrop=drops[2], dropoute=drops[3], dropouth=drops[4])

learner.metrics = [accuracy]
learner.freeze_to(-1)

learner.model.load_state_dict(wgts)

lr=1e-3
lrs = lr



# Run one epoch of fine-tuning
learner.fit(lrs/2, 1, wds=wd, use_clr=(32,2), cycle_len=1)


# Save the fine-tuned model and unfreeze everything to later fine-tune the whole model
learner.save('lm_last_ft')
learner.load('lm_last_ft')
learner.unfreeze()


learner.lr_find(start_lr=lrs/10, end_lr=lrs*10, linear=True)


learner.sched.plot()


# Run this if you want to highly tune the LM to the Amazon data, with 15 epochs
# use_clr controls the shape of the cyclical (triangular) learning rate
learner.fit(lrs, 1, wds=wd, use_clr=(20,10), cycle_len=2)  # cycle_len = 15


# Save the Backbone for further classification!!
learner.save('lm1')
learner.save_encoder('lm1_enc')

learner.sched.plot_loss()
