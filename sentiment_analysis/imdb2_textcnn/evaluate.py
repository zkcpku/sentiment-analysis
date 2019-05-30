from parameters import *
from model import Net
from dataloader import vocab_size,train_iter,valid_iter,clean_string,generate_vocab,encoder_strings,pad_samples
from pretrained_embed import weight
model_path = opt.model_path


net = Net(vocab_size + 1, opt.embed_size, opt.max_len,opt.labels, weight)

checkpoint = torch.load(model_path)

net.load_state_dict(checkpoint['net'])
if opt.evaluate_use_cuda:
	device = torch.device("cuda")
	net.to(device)

def evaluate_train(data_iter):
	net.eval()
	n = len(data_iter.dataset)
	# m = len(valid_iter.dataset)

	data_acc = 0
	with torch.no_grad():
		for feature, label in data_iter:
			if opt.evaluate_use_cuda:
				feature = feature.cuda()
				label = label.cuda()

			score = net(feature)

			data_acc += (torch.argmax(score,dim = 1) == label).sum().item()


	return data_acc,n

def analysis_str(string,debug = False):
	cleaned_string = clean_string(string)
	vocab,word2idx,idx2word = generate_vocab()
	clean_strings = [cleaned_string]
	input_list = pad_samples(encoder_strings(clean_strings,word2idx))

	input_tensor = torch.tensor(input_list)


	if opt.evaluate_use_cuda:
		input_tensor = input_tensor.cuda()

	score = net(input_tensor)
	soft_score = F.softmax(score,dim = 1).cpu().detach().numpy().tolist()[0]
	# print(soft_score[0])

	rst_string = ""
	for e in input_list[0]:
		rst_string += (idx2word[e] + " ")

	if debug:
		for e in input_list[0]:
			print(idx2word[e],end = " ")
		print("\n")
		
	rst = torch.argmax(score,dim = 1).item()
	return rst,soft_score,rst_string

def generate_csv(test_path,save_path):
	test_csv = pd.read_csv(test_path,sep = "\t")
	phrase_id = test_csv['PhraseId'].values
	strings = test_csv['Phrase']

	
	# print(strings[0])


	cleaned_strings = [clean_string(s) for s in strings]
	vocab,word2idx,idx2word = generate_vocab()
	input_list = pad_samples(encoder_strings(cleaned_strings,word2idx))
	input_tensor = torch.tensor(input_list)
	zero_label = torch.zeros(input_tensor.shape[0])
	input_set = torch.utils.data.TensorDataset(input_tensor,zero_label)
	input_iter = torch.utils.data.DataLoader(input_set,batch_size = opt.batch_size, shuffle = False)

	rst = []
	net.eval()
	with torch.no_grad():
		for feature,_ in input_iter:

			if opt.evaluate_use_cuda:
				feature = feature.cuda()
			score = net(feature)
			this_rst = torch.argmax(score, dim = 1).cpu().numpy().tolist()

			rst += this_rst

	rst = np.array(rst,dtype = int)
	final_answer = pd.DataFrame({'PhraseId':phrase_id,'Sentiment':rst})

	final_answer.to_csv(save_path,index = False)





if __name__ == '__main__':
	# train_acc,n = evaluate_train(train_iter)
	# valid_acc,m = evaluate_train(valid_iter)

	# # print(train_acc,"/",n)
	# print(valid_acc,"/",m)
	# print(train_acc + valid_acc,"/",n + m,"=",(train_acc + valid_acc) / (n + m))



	print(analysis_str("I love you."))



	# generate_csv(opt.test_path, opt.save_path)