import sys
import warnings

warnings.filterwarnings("ignore")  # ignore warnings in this notebook

import numpy as np
import torch

from tqdm import *
from tts.hparams import HParams as hp
from tts.audio import save_to_wav
from tts.models import Text2Mel, SSRN
from tts.datasets.lj_speech import vocab, idx2char, get_test_data

torch.set_grad_enabled(False)
text2mel = Text2Mel(vocab)
text2mel.load_state_dict(torch.load("tts/ljspeech-text2mel.pth").state_dict())
text2mel = text2mel.eval()
ssrn = SSRN()
ssrn.load_state_dict(torch.load("tts/ljspeech-ssrn.pth").state_dict())
ssrn = ssrn.eval()

SENTENCES = [
    "The birch canoe slid on the smooth planks.",
    "Glue the sheet to the dark blue background.",
    "It's easy to tell the depth of a well.",
    "These days a chicken leg is a rare dish.",
    "Rice is often served in round bowls.",
    "The juice of lemons makes fine punch.",
    "The box was thrown beside the parked truck.",
    "The hogs were fed chopped corn and garbage.",
    "Four hours of steady work faced us.",
    "Large size in stockings is hard to sell.",
    "The boy was there when the sun rose.",
    "A rod is used to catch pink salmon.",
    "The source of the huge river is the clear spring.",
    "Kick the ball straight and follow through.",
    "Help the woman get back to her feet.",
    "A pot of tea helps to pass the evening.",
    "Smoky fires lack flame and heat.",
    "The soft cushion broke the man's fall.",
    "The salt breeze came across from the sea.",
    "The girl at the booth sold fifty bonds."
]

# synthetize by one by one because there is a batch processing bug!
def englishSynthesizer(sentence:str):
    normalized_sentence = "".join([c if c.lower() in vocab else '' for c in sentence])
    print(normalized_sentence)

    sentences = [normalized_sentence]
    max_N = len(normalized_sentence)
    L = torch.from_numpy(get_test_data(sentences, max_N))
    zeros = torch.from_numpy(np.zeros((1, hp.n_mels, 1), np.float32))
    Y = zeros
    A = None

    for t in range(hp.max_T):
        _, Y_t, A = text2mel(L, Y, monotonic_attention=True)
        Y = torch.cat((zeros, Y_t), -1)
        _, attention = torch.max(A[0, :, -1], 0)
        attention = attention.item()
        if L[0, attention] == vocab.index('E'):  # EOS
            break

    _, Z = ssrn(Y)

    Z = Z.cpu().detach().numpy()
    save_to_wav(Z[0, :, :].T, 'sample.wav')
    #IPython.display.display(Audio('%d.wav' % (i + 1), rate=hp.sr))

englishSynthesizer(sentence="The birch canoe slid on the smooth planks. Glue the sheet to the dark blue background. It's easy to tell the depth of a well.")