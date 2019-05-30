# from pretrained_embed import weight
from parameters import *
from dataloader import vocab_size,generate_vocab





def generate_weight():
	weight = torch.zeros(vocab_size+1, opt.embed_size)
	wvmodel = gensim.models.KeyedVectors.load_word2vec_format(opt.weight_path,
															  binary=False, encoding='utf-8')
	vocab, word2idx, idx2word = generate_vocab()

	not_in = 0

	for i in range(len(wvmodel.index2word)):
		try:
			index = word_to_idx[wvmodel.index2word[i]]
		except:
			not_in += 1
			print(wvmodel.index2word[i])
			continue
		weight[index, :] = torch.from_numpy(wvmodel.get_vector(
			idx_to_word[word_to_idx[wvmodel.index2word[i]]]))

	print(not_in)
	return weight


weight = generate_weight()