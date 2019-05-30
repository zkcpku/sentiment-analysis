from parameters import *
from model import *
from dataloader import SenTree,trn,dev,vocab_size


from nltk.parse.corenlp import CoreNLPParser
from nltk import Tree
from functools import reduce
from nltk.treeprettyprinter import TreePrettyPrinter
from nltk.draw.tree import TreeView

model_path = opt.model_path


net = TreeLSTM(vocab_size)
checkpoint = torch.load(model_path)

net.load_state_dict(checkpoint['net'])

if opt.evaluate_use_cuda:
    device = torch.device("cuda")
    net.to(device)

'''
start dataloader
'''
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
    def getTree(tree, vocabIndicesMapFile =opt.vocab_path):
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
                
    def index2str(tree, vocabIndicesMapFile = opt.vocab_path):
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


parser = CoreNLPParser(url='http://localhost:8000')
'''
ending dataloader
'''

def evaluate_train(data_iter):
    net.eval()
    data_correctRoot,data_correctAll = net.evaluate(data_iter)
    print("correctRoot :",data_correctRoot)
    print("correctAll  :",data_correctAll)

    return data_correctRoot,data_correctAll


def showInput(my_sentence,draw_pic = False):
    t, = parser.raw_parse(my_sentence)
    # t.draw()
    bt = binarize(t)
    # bt.draw()
    tree = bt.pformat()
    input = SenTreeTest.getTree(tree)
    # input.draw()
    net.eval()

    # print(type(dev[0]))
    # print(type(input))
    predictions, loss = net.getLoss(input)
    scores = net.forward(input).cpu().detach().numpy()
    scores_exp = np.exp(scores)
    possi = scores_exp / scores_exp.sum(axis = 1).reshape(-1,1)
    root_possi = possi[-1]
    root_class = np.argmax(root_possi)
    # print((predictions.data))
    # print((net.labelList.data))
    pred = predictions.data
    label = net.labelList.data
    index2scores = {}
    for i in range(len(pred)):
        p = pred[i].item()
        p_list = ["very negative","negative","neutral","positive","very positive"]
#         p_list = ["--","-","0","+","++"]
        index2scores[label[i].item()] = p_list[p]
    for subtree in input.subtrees():
        i = subtree.label()
        subtree.set_label(index2scores[i])
    input.index2str()
    if draw_pic:
        input.draw()
        input.pretty_print()
    outimage = TreePrettyPrinter(input)
    outimage = str(outimage)


    return outimage,root_possi,root_class


def generate_csv(test_path,save_path):
    test_csv = pd.read_csv(test_path,sep = "\t")
    phrase_id = test_csv['PhraseId'].values
    strings = test_csv['Phrase']

    rst = []
    net.eval()
    for e in strings:
        rst.append(showInput(e)[2])

    rst = np.array(rst, dtype = int)

    final_answer = pd.DataFrame({'PhraseId':phrase_id,'Sentiment':rst})

    final_answer.to_csv(save_path,index = False)


if __name__ == '__main__':
    # evaluate_train(trn)
    # evaluate_train(dev)
    # generate_csv("test.tsv","rst.csv")
    tmp = showInput("I love you.")
    print(tmp[0])
    print(tmp[1])
    print(tmp[2])