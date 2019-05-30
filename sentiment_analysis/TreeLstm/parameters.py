import sys
import random
import progressbar

from nltk.tree import ParentedTree
import _pickle as cPickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torch.autograd as autograd
# import torchtext.vocab as torchvocab
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_

from torch.optim.lr_scheduler import ReduceLROnPlateau

class Opt(object):
	"""
	describe the parameters
	"""
	def __init__(self):
		# dataloader.py
		# self.root_path = ".."
		self.root_path = "E:/P大课件/Sentiment/rst"
		self.train_path = self.root_path + "/data/trees/train.txt"
		self.small_path = self.root_path + "/data/trees/small.txt"
		self.dev_path = self.root_path + "/data/trees/dev.txt"
		self.test_path = self.root_path + "/data/trees/test.txt"

		self.vocab_path = self.root_path+"/data/trees/train.vocab"

		# train.py
		self.lr = 0.001
		self.use_cuda = True
		self.save_cp = 10
		self.num_epochs = 100

		self.model_root = "E:/P大课件/Sentiment/rst/TreeLstm/"
		self.model_path = self.model_root + "60.pickle"
		self.evaluate_use_cuda = True


		self.widgets = [progressbar.Percentage(), ' ', progressbar.Bar(), ' ', progressbar.ETA()]






opt = Opt()

if __name__ == '__main__':
	print(opt.lr)
