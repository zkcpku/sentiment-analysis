import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torch.autograd as autograd
# import torchtext.vocab as torchvocab
from torch.autograd import Variable

import pickle

import pandas as pd
import numpy as np

import tqdm
import os
import time
import re
import pandas as pd
import string
import gensim
import time
import random
# import snowballstemmer
import collections
from collections import Counter

# from nltk.corpus import stopwords
# 数据预处理使用
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from itertools import chain


# from sklearn.metrics import accuracy_score


# https://samaelchen.github.io/pytorch_lstm_sentiment/

class Opt(object):
	"""
	describe the parameters
	"""
	def __init__(self):
		# dataloader.py
		# self.root_path = ".."
		self.root_path = "E:/P大课件/Sentiment/rst"
		self.train_path = self.root_path+"/data/train.tsv"
		self.test_path = self.root_path+"/data/test.tsv"

		self.max_len = 48
		self.use_old_vocab = True
		self.vocab_path = self.root_path+"/data/vocab.pickle"

		self.use_old_dataloader = True
		self.dataloader_path = self.root_path+"/data/dataloader.pickle"
		self.valid_per = 0.2

		self.train_shuffle = True
		
		self.batch_size = 256
		# train.py
		self.lr = 0.01
		self.weight_decay = 2e-5
		self.use_cuda = True


		self.num_epochs = 100
		self.save_cp = 10


		# model.py
		self.use_embed = False
		self.embed_size = 100
		self.num_hiddens = 100
		self.num_layers = 2
		self.bidirectional = False
		self.labels = 5

		# evaluate.py
		self.model_root = "E:/P大课件/Sentiment/rst/textCNN/"
		self.model_path = self.model_root + "5_26cnn_weigthdecay.pickle"
		self.save_path = self.model_root + "rst.csv"
		self.evaluate_use_cuda = True


opt = Opt()

if __name__ == '__main__':
	print(opt.lr)