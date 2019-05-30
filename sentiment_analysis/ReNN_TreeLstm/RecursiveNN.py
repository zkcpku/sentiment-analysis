import sys
import random
import progressbar
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm
from SenTree import *

class RecursiveNN(nn.Module):
    def __init__(self, vocabSize, embedSize=100, numClasses=5):
        super(RecursiveNN, self).__init__()
        self.embedding = nn.Embedding(int(vocabSize), embedSize)
        self.W = nn.Linear(2*embedSize, embedSize, bias=True)
        self.projection = nn.Linear(embedSize, numClasses, bias=True) # 对每个节点进行五分类的预测，将其softmax即为各个种类的概率
        self.activation = F.relu
        self.nodeProbList = [] # 用来存储各个节点的概率值
        self.labelList = [] # 用来存储各个节点的正确值

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
        '''
        返回 预测值 和 loss(使用交叉熵)
        '''
        nodes = self.forward(tree)
        predictions = nodes.max(dim=1)[1]
        loss = F.cross_entropy(input=nodes, target=self.labelList)
        return predictions,loss

    def evaluate(self, trees):
        '''
        评估函数
        correctAll 为节点 的预测正确的个数
        correctRoot 为根节点 的预测正确的个数
        n 为总的树的个数

        返回 平均每棵树的 正确节点数, 正确根节点数
        '''
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

CUDA=True
if len(sys.argv)>1:
    if sys.argv[1].lower()=="cuda": CUDA=True

print("Reading and parsing trees")
trn = SenTree.getTrees("./trees/train.txt","train.vocab")
dev = SenTree.getTrees("./trees/dev.txt",vocabIndicesMapFile="train.vocab")

if CUDA: model = RecursiveNN(SenTree.vocabSize).cuda()
else: model = RecursiveNN(SenTree.vocabSize)
max_epochs = 100
widgets = [progressbar.Percentage(), ' ', progressbar.Bar(), ' ', progressbar.ETA()]

optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, dampening=0.0)
bestAll=bestRoot=0.0
for epoch in range(max_epochs):
  print("Epoch %d" % epoch)
  pbar = progressbar.ProgressBar(widgets=widgets, maxval=len(trn)).start()
  for step, tree in enumerate(trn):
    predictions, loss = model.getLoss(tree)
    optimizer.zero_grad()
    loss.backward()
    clip_grad_norm(model.parameters(), 5, norm_type=2.)
    optimizer.step()
    pbar.update(step)
  pbar.finish()
  correctRoot, correctAll = model.evaluate(dev)
  if bestAll<correctAll: bestAll=correctAll
  if bestRoot<correctRoot: bestRoot=correctRoot
  print("\nValidation All-nodes accuracy:"+str(correctAll)+"(best:"+str(bestAll)+")")
  print("Validation Root accuracy:" + str(correctRoot)+"(best:"+str(bestRoot)+")")
  random.shuffle(trn)