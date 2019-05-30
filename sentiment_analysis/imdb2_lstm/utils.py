from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

def trans_glove_format(input_path,output_path):
	# 输入文件
	glove_file = datapath(input_path)
	# 输出文件
	tmp_file = get_tmpfile(output_path)

	# call glove2word2vec script
	# default way (through CLI): python -m gensim.scripts.glove2word2vec --input <glove_file> --output <w2v_file>

	# 开始转换
	glove2word2vec(glove_file, tmp_file)

	# 加载转化后的文件
	model = KeyedVectors.load_word2vec_format(tmp_file)
	return model


if __name__ == '__main__':
	model = trans_glove_format("E:/P大课件/Sentiment/rst/data/glove.twitter.27B.100d.txt", "E:/P大课件/Sentiment/rst/data/glove_word2vec_format.txt")