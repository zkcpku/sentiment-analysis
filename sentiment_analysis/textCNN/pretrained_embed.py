# from pretrained_embed import weight
from parameters import *
from dataloader import vocab_size

weight = torch.zeros(vocab_size+1, opt.embed_size)