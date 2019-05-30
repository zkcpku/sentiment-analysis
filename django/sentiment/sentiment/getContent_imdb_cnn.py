import sys
sys.path.append("E:/P大课件/Sentiment/rst")
sys.path.append("E:/P大课件/Sentiment/rst/imdb2_textcnn")
from imdb2_textcnn.evaluate import analysis_str as cnn_api



def toStr(string):
	score = cnn_api(string)
	rst = "imdb_textcnn_glove:   " + string  + "\n"
	rst = rst + "cleaned string:" + "\n"+score[2] + "\n"
	rst = rst + "sentiment score:" + str(score[0])
	rst = rst + " \n " + "softmax score(each class):" + " \n "
	for i in range(len(score[1])):
		rst = rst + "class "+str(i)+":\t"+str(score[1][i]) + " \n "

	return rst


if __name__ == '__main__':
	print(toStr("I love you."))

