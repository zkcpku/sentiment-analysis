from parameters import *
lemmatizer = WordNetLemmatizer()

# 修改此处数据地址
with open(opt.train_path) as f:
	train_csv = pd.read_csv(f,sep = "\t")
# train_csv = pd.read_csv(opt.train_path,sep = "\t")
with open(opt.test_path) as f:
	test_csv = pd.read_csv(f,sep = "\t")
# test_csv = pd.read_csv(opt.test_path,sep = "\t")
# print(train_csv.head())
# print(test_csv.head())

def clean_string(string):
	'''
	#remove non-alphabetic characters
	#tokenize the sentences
	#lemmatize each word to its lemma

	return list [words]
	'''
	string = re.sub("[^a-zA-Z]"," ", string)
	words = word_tokenize(string.lower())
	words_lem = [lemmatizer.lemmatize(w) for w in words]
	
	return words_lem

def generate_vocab(use_old = True):
	if use_old:
		with open(opt.vocab_path,"rb") as f:
			vocab, word_to_idx, idx_to_word = pickle.load(f)
	else:
		train_phrases = train_csv['Phrase']
		phrases_cleaned = [clean_string(s) for s in train_phrases]
		# train_tokens = [w for w in s for s in phrases_cleaned]
		vocab = set(chain(*phrases_cleaned))
		
		word_to_idx  = {word: i+1 for i, word in enumerate(vocab)}
		word_to_idx['<unk>'] = 0
		idx_to_word = {i+1: word for i, word in enumerate(vocab)}
		idx_to_word[0] = '<unk>'
		with open(opt.vocab_path, "wb") as f:
			pickle.dump((vocab,word_to_idx,idx_to_word), f)

	vocab_size = len(vocab)
	return vocab, word_to_idx, idx_to_word

def encoder_strings(strings,word_to_idx,UNK = 0):
	features = []
	for string in strings:
		feature = []
		for token in string:
			if token in word_to_idx:
				feature.append(word_to_idx[token])
			else:
				feature.append(UNK)
		features.append(feature)

	return features

def pad_samples(features, maxlen=48, PAD=0):
	'''
	pad the setences with PAD = 0
	'''
	padded_features = []
	for feature in features:
		if len(feature) >= maxlen:
			padded_feature = feature[:maxlen]
		else:
			padded_feature = feature
			while(len(padded_feature) < maxlen):
				padded_feature.append(PAD)
		padded_features.append(padded_feature)
	return padded_features


def generate_set(valid_per = 0.2):
	'''
	return train_iter,valid_iter

	torch.utils.data.DataLoader
	'''

	if opt.use_old_dataloader:
		with open(opt.dataloader_path,"rb") as f:
			train_set,valid_set = pickle.load(f)
	else:
		vocab,word2idx,idx2word = generate_vocab(use_old = opt.use_old_vocab)

		train_set = [clean_string(w) for w in train_csv['Phrase']]
		label_set = [s for s in train_csv['Sentiment']]
		train_features_list = pad_samples(encoder_strings(train_set,word2idx))
		train_total_list = [(train_features_list[i], label_set[i]) for i in range(len(label_set))]

		random.shuffle(train_total_list)

		split_num = int((1-valid_per) * len(train_total_list))

		train_list = train_total_list[:split_num]
		valid_list = train_total_list[split_num:]

		train_features = torch.tensor([p for p,s in train_list])
		train_labels = torch.tensor([s for p,s in train_list])

		valid_features = torch.tensor([p for p,s in valid_list])
		valid_labels = torch.tensor([s for p,s in valid_list])

		train_set = torch.utils.data.TensorDataset(train_features, train_labels)
		valid_set = torch.utils.data.TensorDataset(valid_features,valid_labels)

		with open(opt.dataloader_path,"wb") as f:
			pickle.dump((train_set,valid_set), f)

	return train_set,valid_set

def generate_iter():
	train_set,valid_set = generate_set(opt.valid_per)
	train_iter = torch.utils.data.DataLoader(train_set,batch_size = opt.batch_size, shuffle = opt.train_shuffle)
	valid_iter = torch.utils.data.DataLoader(valid_set,batch_size = opt.batch_size, shuffle = False)

	return train_iter,valid_iter


train_iter,valid_iter = generate_iter()
vocab_size = len(generate_vocab()[0])

if __name__ == '__main__':
	vocab,word2idx,idx2word = generate_vocab()
	print(len(vocab))
	print(len(train_iter))
	print(len(valid_iter))

	for p,s in train_iter:
		for e in p[0]:
			# print(e.item())
			print(idx2word[e.item()],end = " ")
		print()
		print(s[0].item())
		break

