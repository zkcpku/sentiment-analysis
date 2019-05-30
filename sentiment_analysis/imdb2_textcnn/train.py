from parameters import *
from model import Net
from dataloader import vocab_size,train_iter,valid_iter

if opt.use_embed:
	from pretrained_embed import weight
else:
	weight = None

device = torch.device("cuda") if opt.use_cuda else torch.device("cpu")

# 训练的三个关键变量
net = Net(vocab_size + 1, opt.embed_size, opt.max_len,opt.labels, weight)
net.to(device)
optimizer = optim.Adam(net.parameters(),lr = opt.lr,weight_decay = opt.weight_decay)
epoch = 0

loss_func = nn.CrossEntropyLoss()

def save_checkpoint(path):
	state = {'net':net.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch}
	torch.save(state, path)

def load_checkpoint(path):
	checkpoint = torch.load(path)
	net.load_state_dict(checkpoint['net'])
	optimizer.load_state_dict(checkpoint['optimizer'])
	epoch = checkpoint['epoch'] + 1


def train(num_epochs):
	n = len(train_iter.dataset)
	m = len(valid_iter.dataset)
	print(n,m)
	for e in range(epoch,num_epochs):
		train_loss = 0
		train_acc = 0
		valid_acc = 0
		valid_loss = 0
		net.train()
		for feature, label in train_iter:
			net.zero_grad()
			
			if opt.use_cuda:
				feature = feature.cuda()
				label = label.cuda()

			# print(feature.shape)
			score = net(feature)
			# print(score.shape)

			loss = loss_func(score,label)
			loss.backward()
			# print(feature.shape)
			# print(label.shape)
			# print(loss.item())
			optimizer.step()

			# print(score.shape)
			# print(score[0])
			# print(label.shape)
			predict = (torch.argmax(score,dim = 1) == label)

			train_acc += predict.sum().item()
			# print(train_acc)
			# print(predict.sum().item())

			train_loss += loss.item()


		print("epoch ",e,":")
		print("train_loss: ", train_loss / n)
		print("train_acc: ",train_acc,"/",n," = ",train_acc/n)
		print("- - - - - - - - - - - - - - - - - -")

		net.eval()
		with torch.no_grad():

			for feature,label in valid_iter:
				net.zero_grad()
				if opt.use_cuda:
					feature = feature.cuda()
					label = label.cuda()

				score = net(feature)

				loss = loss_func(score,label)

				valid_acc += (torch.argmax(score,dim = 1) == label).sum().item()

				valid_loss += loss.item()

			print("valid_loss: ", valid_loss / m)
			print("valid_acc: ",valid_acc,"/",m," = ",valid_acc/m)
			print("-----------------------------------")

		if (e+1) % opt.save_cp == 0:
			save_checkpoint(str(e+1) + ".pickle")


if __name__ == '__main__':
	train(opt.num_epochs)
	save_checkpoint("5_26.pickle")