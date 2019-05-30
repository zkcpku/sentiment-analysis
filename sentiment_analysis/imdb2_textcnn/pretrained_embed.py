# from pretrained_embed import weight
from parameters import *
from dataloader import vocab_size,generate_vocab





def generate_weight(use_old = True):
	if use_old:
		with open(opt.weight_path,"rb") as f:
			weight = pickle.load(f)
	else:
		weight = torch.zeros(vocab_size+1, opt.embed_size)
		wvmodel = gensim.models.KeyedVectors.load_word2vec_format(opt.glove_path,
																  binary=False, encoding='utf-8')
		vocab, word2idx, idx2word = generate_vocab()
		in_sum = 0
		# not_in = 0

		for i in range(len(wvmodel.index2word)):
			# if wvmodel.index2word[i] not in word2idx:
			# 	print(wvmodel.index2word[i])
			# 	return weight
			try:
				index = word2idx[wvmodel.index2word[i]]
				in_sum += 1
			except:
				# not_in += 1
				# print(wvmodel.index2word[i])
				continue
			weight[index, :] = torch.from_numpy(wvmodel.get_vector(
				idx2word[word2idx[wvmodel.index2word[i]]]))
		print(len(word2idx),"/",in_sum)

		with open(opt.weight_path,"wb") as f:
			pickle.dump(weight, f)


	return weight


weight = generate_weight(opt.use_old_weight)

# if __name__ == '__main__':
# 	weight = generate_weight(False)
# 	