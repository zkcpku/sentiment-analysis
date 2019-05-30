import sys
import random
import progressbar
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from SenTree import *
from torch.optim.lr_scheduler import ReduceLROnPlateau

from nltk.tree import ParentedTree
import _pickle as cPickle

from nltk.parse.corenlp import CoreNLPParser
from nltk import Tree
from functools import reduce
from nltk.treeprettyprinter import TreePrettyPrinter

from nltk.draw.tree import TreeView

import numpy as np

import matplotlib.pyplot as plt

class RecursiveNN(nn.Module):
    def __init__(self, vocabSize, embedSize=100, numClasses=5):
        super(RecursiveNN, self).__init__()
        self.embedding = nn.Embedding(int(vocabSize), embedSize)
        self.W = nn.Linear(2*embedSize, embedSize, bias=True)
        self.projection = nn.Linear(embedSize, numClasses, bias=True) # 对每个节点进行五分类的预测，将其softmax即为各个种类的概率
        self.activation = nn.ReLU()
        self.nodeProbList = [] # 用来存储各个节点的概率值
        self.labelList = [] # 用来存储各个节点的正确值
        self.crossentropy = nn.CrossEntropyLoss()

    def traverse(self, node):
        '''
        用来递归地获取每个节点的概率值
        并保存在nodeProbList
        并将对应的label值存在labelList中
        返回输入node的激活值
        '''
        if node.isLeaf(): currentNode = self.activation(self.embedding(Var(torch.LongTensor([node.getLeafWord()])))) 
        # 对于叶节点，直接计算embedding后的激活值，即f(a)
        else: currentNode = self.activation(self.W(torch.cat((self.traverse(node.left()),self.traverse(node.right())),1)))
        # 否则将左右节点连接(cat)，在经过一个线性层，即f(W * [a b])，相当于这里的父节点的embedding为[a b]
        self.nodeProbList.append(self.projection(currentNode))
        self.labelList.append(torch.LongTensor([node.label()]))
        return currentNode

    def forward(self, x):
        '''
        前向传播 返回各个节点的预测值
        '''
        self.nodeProbList = []
        self.labelList = []
        self.traverse(x)
        self.labelList = Var(torch.cat(self.labelList))
        return torch.cat(self.nodeProbList)

    def getLoss(self, tree):
        nodes = self.forward(tree)
        predictions = nodes.max(dim=1)[1]
        loss = self.crossentropy(input = nodes, target = self.labelList)
#         loss = F.cross_entropy(input=nodes, target=self.labelList)
        return predictions,loss

    def evaluate(self, trees):
        pbar = progressbar.ProgressBar(widgets=widgets, maxval=len(trees)).start()
        n = nAll = correctRoot = correctAll = 0.0
        for j, tree in enumerate(trees):
            predictions,loss = self.getLoss(tree)
            correct = (predictions.data==self.labelList.data)
            correctAll += correct.sum()
            nAll += correct.squeeze().size()[0]
            correctRoot += correct.squeeze()[-1]
            n += 1
            pbar.update(j)
        pbar.finish()
        return correctRoot.item() / n, correctAll.item() /nAll

def Var(v):
    if CUDA: return Variable(v.cuda())
    else: return Variable(v)
    
# 使用save保存模型，并转换到cpu上保存，使用的时候在转换到gpu上
def save_model(model, filename):
    state = model.state_dict()
    for key in state: state[key] = state[key].clone().cpu()
    torch.save(state, filename)

class TreeLSTM(nn.Module):
    def __init__(self, vocabSize, hdim=100, numClasses=5):
        super(TreeLSTM, self).__init__()
        self.embedding = nn.Embedding(int(vocabSize), hdim)
        self.Wi = nn.Linear(hdim, hdim, bias=True)
        self.Wo = nn.Linear(hdim, hdim, bias=True)
        self.Wu = nn.Linear(hdim, hdim, bias=True)
        self.Ui = nn.Linear(2 * hdim, hdim, bias=True)
        self.Uo = nn.Linear(2 * hdim, hdim, bias=True)
        self.Uu = nn.Linear(2 * hdim, hdim, bias=True)
        self.Uf1 = nn.Linear(hdim, hdim, bias=True)
        self.Uf2 = nn.Linear(hdim, hdim, bias=True)
        self.projection = nn.Linear(hdim, numClasses, bias=True)
        self.activation = nn.ReLU()
        self.nodeProbList = []
        self.labelList = []
        self.crossentropy = nn.CrossEntropyLoss()
        

    def traverse(self, node):
        if node.isLeaf():
            e = self.embedding(Var(torch.LongTensor([node.getLeafWord()])))
            i = torch.sigmoid(self.Wi(e))
            o = torch.sigmoid(self.Wo(e))
            u = self.activation(self.Wu(e))
            c = i * u
        else:
            leftH,leftC = self.traverse(node.left())
            rightH,rightC = self.traverse(node.right())
            e = torch.cat((leftH, rightH), 1)
            i = torch.sigmoid(self.Ui(e))
            o = torch.sigmoid(self.Uo(e))
            u = self.activation(self.Uu(e))
            c = i * u + torch.sigmoid(self.Uf1(leftH)) * leftC + torch.sigmoid(self.Uf2(rightH)) * rightC
        h = o * self.activation(c)
        self.nodeProbList.append(self.projection(h))
        self.labelList.append(torch.LongTensor([node.label()]))
        return h,c

    def forward(self, x):
        self.nodeProbList = []
        self.labelList = []
        self.traverse(x)
        self.labelList = Var(torch.cat(self.labelList))
        return torch.cat(self.nodeProbList)

    def getLoss(self, tree):
        nodes = self.forward(tree)
        predictions = nodes.max(dim=1)[1]
        loss = self.crossentropy(input=nodes, target=self.labelList)
        return predictions,loss

    def evaluate(self, trees):
        pbar = progressbar.ProgressBar(widgets=widgets, maxval=len(trees)).start()
        n = nAll = correctRoot = correctAll = 0.0
        for j, tree in enumerate(trees):
            predictions,loss = self.getLoss(tree)
            correct = (predictions.data==self.labelList.data)
            correctAll += correct.sum()
            nAll += correct.squeeze().size()[0]
            correctRoot += correct.squeeze()[-1]
            n += 1
            pbar.update(j)
        pbar.finish()
        return correctRoot.item() / n, correctAll.item()/nAll

class SenTreeTest(ParentedTree):
    def __init__(self, node, children=None):
        super(SenTreeTest,self).__init__(node, children)

    def left(self):
        return self[0]

    def right(self):
        return self[1]

    def isLeaf(self):
        return self.height()==2

    def getLeafWord(self):
        return self[0]
    def getTree(tree, vocabIndicesMapFile ="train.vocab"):
        tree = SenTreeTest.fromstring(tree)
        vocabIndicesMap=cPickle.load(open(vocabIndicesMapFile,'rb'))
        SenTreeTest.mapTreeNodes(tree,vocabIndicesMap)
        index = 0
        for subtree in tree.subtrees():
            subtree.set_label(index)
            index += 1
        return tree
    def mapTreeNodes(tree, vocabIndicesMap):
        for leafPos in tree.treepositions('leaves'):
            if tree[leafPos] in vocabIndicesMap: tree[leafPos] = vocabIndicesMap[tree[leafPos]]
            else: tree[leafPos]= vocabIndicesMap['UNK']
                
    def index2str(tree, vocabIndicesMapFile = "train.vocab"):
        index2str = {}
        vocabIndicesMap=cPickle.load(open(vocabIndicesMapFile,'rb'))
        for k in vocabIndicesMap:
            index2str[vocabIndicesMap[k]] = k
        for leafPos in tree.treepositions('leaves'):
            tree[leafPos] = index2str[tree[leafPos]]
        return tree


def binarize(tree):
    """
    Recursively turn a tree into a binary tree.
    """
    if isinstance(tree, str):
        return Tree('0',[tree])
    elif len(tree) == 1:
#         print(tree)
#         print('\n')
        return binarize(tree[0])
    else:
        label = tree.label()
#         print(type(label))
        return reduce(lambda x, y: Tree(label, (binarize(x), binarize(y))), tree)


widgets = [progressbar.Percentage(), ' ', progressbar.Bar(), ' ', progressbar.ETA()]
CUDA=True
if len(sys.argv)>1:
	if sys.argv[1].lower()=="cuda": CUDA=True

print("Reading and parsing trees")
	# trn = SenTree.getTrees("./trees/train.txt","train.vocab") # 第一次解析的时候需要生成词向量
trn = SenTree.getTrees("./trees/train.txt",vocabIndicesMapFile="train.vocab") # 修改后
dev = SenTree.getTrees("./trees/dev.txt",vocabIndicesMapFile="train.vocab")

use_old_model = input("use old model?(y)")
if use_old_model == 'y':
	model = TreeLSTM(SenTree.vocabSize)
	model_name = input()
	#     model_name = 'model/' + model_name + '.model'
	model.load_state_dict(torch.load('model/' + model_name + '.model'))
	model = model.cuda()
	correctRoot, correctAll = model.evaluate(dev)
	print(correctRoot)
	print(correctAll)
	#     optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, dampening=0.0)
	#     optimizer.load_state_dict(torch.load('model/opt_'+ model_name + '.opt'))

parser = CoreNLPParser(url='http://localhost:8000')

def getSentimentTree(sentence):
	try:
		# my_sentence = "I love you."
		print("get Sentence: " + sentence)
		my_sentence = sentence
		t, = parser.raw_parse(my_sentence)
		# t.draw()
		bt = binarize(t)
		# bt.draw()
		tree = bt.pformat()
		input = SenTreeTest.getTree(tree)
		# input.draw()
		model.eval()

		predictions, loss = model.getLoss(input)
		# print((predictions.data))
		# print((model.labelList.data))
		pred = predictions.data
		label = model.labelList.data
		index2scores = {}
		for i in range(len(pred)):
		    p = pred[i].item()
		#     p_list = ["very negative","negative","neutral","positive","very positive"]
		    p_list = ["--","-","0","+","++"]
		    index2scores[label[i].item()] = p_list[p]
		#     index2scores[label[i].item()] = p
		for subtree in input.subtrees():
		    i = subtree.label()
		    subtree.set_label(index2scores[i])
		input.index2str()
		# TreeView(input)
		# TreeView(input)._cframe.print_to_file('output.ps')
		# https://stackoverflow.com/questions/23429117/saving-nltk-drawn-parse-tree-to-image-file
		# https://stackoverflow.com/questions/44880337/use-tkinter-for-nltk-draw-inside-of-jupyter-notebook
		# import os
		# os.system('magick output.ps output.png')
		# from IPython.display import Image
		# Image(filename='output.png')
		# input.pretty_print()
		outimage = TreePrettyPrinter(input)
		# text2png(outimage, 'test.png')
		outimage = str(outimage)
		return outimage
		# print(outimage)	
	except Exception as e:
		print(e)
		return "Error!Please contact with the author."


def getRootGraph(sentence):
	try:
		# my_sentence = "I love you."
		print("get Sentence: " + sentence)
		my_sentence = sentence
		t, = parser.raw_parse(my_sentence)
		# t.draw()
		bt = binarize(t)
		# bt.draw()
		tree = bt.pformat()
		input = SenTreeTest.getTree(tree)
		# input.draw()
		model.eval()
		scores = model.forward(input).cpu()
		scores = scores.detach().numpy()

		scores_exp = np.exp(scores)
		possi = scores_exp / scores_exp.sum(axis=1).reshape(-1,1)
		# print(possi)
		plt.bar(['--','-','0','+','++'],possi[-1])
		plt.title(my_sentence)
		filename = "root_graph.png"
		plt.savefig(filename)
		# plt.show()

		# print(type(dev[0]))
		# print(type(input))
		return filename
		# print(outimage)	
	except Exception as e:
		print(e)
		# return "Error!Please contact with the author."
