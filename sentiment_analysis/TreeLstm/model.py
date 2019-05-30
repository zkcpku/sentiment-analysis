from parameters import *
from dataloader import *

def Var(v):
    if opt.use_cuda: return Variable(v.cuda())
    else: return Variable(v)

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
            i = torch.sigmoid(self.Ui(e)) # 决定是否遗忘
            o = torch.sigmoid(self.Uo(e)) # 
            u = self.activation(self.Uu(e)) # 生成新的记忆
            c = i * u + torch.sigmoid(self.Uf1(leftH)) * leftC + torch.sigmoid(self.Uf2(rightH)) * rightC # 最终的记忆
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
        widgets = opt.widgets
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


