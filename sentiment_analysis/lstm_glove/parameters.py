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
		self.train_path = self.root_path + "/data/train.tsv"
		self.test_path = self.root_path + "/data/test.tsv"

		self.max_len = 48
		self.use_old_vocab = True
		self.vocab_path = self.root_path + "/data/vocab.pickle"

		self.use_old_dataloader = True
		self.dataloader_path = self.root_path + "/data/dataloader.pickle"
		self.valid_per = 0.2

		self.train_shuffle = True
		
		self.batch_size = 256
		# train.py
		self.lr = 0.01
		self.use_cuda = True

		self.num_epochs = 100
		self.save_cp = 10
		# weight
		self.use_old_weight = True # 使用已保存的词向量文件
		self.glove_path = self.root_path + "/data/glove_word2vec_format.txt"
		self.weight_path = self.root_path + "/data/weight_with_0.pickle"
		# model.py
		self.use_embed = True # 使用词向量
		self.embed_size = 100
		self.num_hiddens = 100
		self.num_layers = 1
		self.bidirectional = False
		self.labels = 5


		# evaluate.py
		self.model_path = "E:/P大课件/Sentiment/rst/lstm_glove/"+"5_26.pickle"
		self.save_path = "rst.csv"
		self.evaluate_use_cuda = True

opt = Opt()

if __name__ == '__main__':
	print(opt.lr)