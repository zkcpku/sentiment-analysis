import sys
sys.path.append("E:/P大课件/Sentiment/rst")
sys.path.append("E:/P大课件/Sentiment/rst/lstm_glove")
from lstm_glove.evaluate import analysis_str as lstm_glove_api



def toStr(string):
	score = lstm_glove_api(string)
	rst = "lstm_glove:   " + string  + "\n"
	rst = rst + "cleaned string:" + "\n"+score[2] + "\n"
	rst = rst + "sentiment score:" + str(score[0])
	rst = rst + " \n " + "softmax score(each class):" + " \n "
	for i in range(len(score[1])):
		rst = rst + "class "+str(i)+":\t"+str(score[1][i]) + " \n "

	return rst


if __name__ == '__main__':
	print(toStr("I love you."))

