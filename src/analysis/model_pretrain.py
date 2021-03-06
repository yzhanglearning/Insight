############## using pretrained language model ##############

###### load the weights of pre-trained model ########
import torch
from os.path import join as Path
import numpy as np
from fastai.text import *

def preTrainModel(pre_lm_path):

    wgts = torch.load(pre_lm_path, map_location=lambda storage, loc: storage)

    # ###??? random initialization weights:
    for w_i in wgts:
        wgts[w_i] = torch.rand(wgts[w_i].shape) ###???

    # check the word embedding layer and keep a "mean word" for unkonw tokens
    enc_wgts = to_np(wgts['0.encoder.weight'])
    row_m = enc_wgts.mean(0)

    print('the encoding weights shape is: ', enc_wgts.shape)
    return wgts, enc_wgts, row_m


def newEmbed(vs, em_sz, wgts, enc_wgts, row_m, itos, pre_path):
    # Load the vocabulary on which the pre-trained model was trained
    # Define an embedding matrix with the vocabulary of our dataset
    f = open(Path(pre_path, 'itos_wt103.pkl'), 'rb')
    itos2 = pickle.load(f)
    stoi2 = collections.defaultdict(lambda: -1, {v:k for k, v in enumerate(itos2)})

    new_w = np.zeros((vs, em_sz), dtype=np.float32)
    for i, w in enumerate(itos):
        r = stoi2[w]
        new_w[i] = enc_wgts[r] if r >= 0 else row_m

    # use the new embedding matrix for the pre-trained models
    wgts['0.encoder.weight'] = T(new_w)
    wgts['0.encoder_with_dropout.embed.weight'] = T(np.copy(new_w))
    wgts['1.decoder.weight'] = T(np.copy(new_w))

    return wgts


################## tuning the embedding layer ###################

def modelTuning(pre_path, wgts, trn_lm, val_lm, vs, em_sz, nh, nl, emb_tune=True, sys_tune=False):
# Define the learner object to do the fine-tuning
# Here we will freeze everything except the embedding layer, so that we can have a better
# embedding for unknown words than just the mean embedding on which we initialise it.
    wd=1e-7
    bptt=70
    bs=52
    opt_fn = partial(optim.Adam, betas=(0.8, 0.99))

    trn_dl = LanguageModelLoader(np.concatenate(trn_lm), bs, bptt)
    val_dl = LanguageModelLoader(np.concatenate(val_lm), bs, bptt)
    md = LanguageModelData(pre_path, 1, vs, trn_dl, val_dl, bs=bs, bptt=bptt)

    drops = np.array([0.25, 0.1, 0.2, 0.02, 0.15])*0.7

    learner= md.get_model(opt_fn, em_sz, nh, nl,
        dropouti=drops[0], dropout=drops[1], wdrop=drops[2], dropoute=drops[3], dropouth=drops[4])

    learner.metrics = [accuracy]
    learner.freeze_to(-1)

    learner.model.load_state_dict(wgts)

    lr=1e-3
    lrs = lr

    if emb_tune:
        # Run one epoch of fine-tuning
        learner.fit(lrs/2, 1, wds=wd, use_clr=(32,2), cycle_len=1)

        # Save the fine-tuned model and unfreeze everything to later fine-tune the whole model
        learner.save('lm_last_ft')
        learner.load('lm_last_ft')
        learner.unfreeze()
    if sys_tune:

        learner.lr_find(start_lr=lrs/10, end_lr=lrs*10, linear=True)
        learner.sched.plot()

        # Run this if you want to highly tune the LM to the Amazon data, with 15 epochs
        # use_clr controls the shape of the cyclical (triangular) learning rate
        learner.fit(lrs, 1, wds=wd, use_clr=(20,10), cycle_len=2)  # cycle_len = 15

        # Save the Backbone for further classification!!
        learner.save('lm1')
        learner.save_encoder('lm1_enc')

        learner.sched.plot_loss()
