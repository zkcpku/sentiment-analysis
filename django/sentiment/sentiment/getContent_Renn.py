import sys
sys.path.append("E:/P大课件/Sentiment/rst")
sys.path.append("E:/P大课件/Sentiment/rst/RennLstm")
from RennLstm.evaluate import showInput as renn_api



def toStr(string):
	score = renn_api(string)
	rst = "RecursiveNN:   " + string  + "\n"
	rst = rst + "tree:" + "\n"+score[0] + "\n"
	rst = rst + "sentiment score:" + str(score[2])
	rst = rst + " \n " + "softmax score(each class):" + " \n "
	for i in range(len(score[1])):
		rst = rst + "class "+str(i)+":\t"+str(score[1][i]) + " \n "

	return rst


if __name__ == '__main__':
	print(toStr("I love you."))

