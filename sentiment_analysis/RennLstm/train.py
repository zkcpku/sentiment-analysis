from parameters import *
from model import RecursiveNN
from dataloader import SenTree,trn,dev,vocab_size

print("vocab_size:",vocab_size)

device = torch.device("cuda") if opt.use_cuda else torch.device("cpu")

# 训练的三个关键变量

net = RecursiveNN(vocab_size)
net.to(device)
optimizer = optim.Adam(net.parameters(),lr = opt.lr)
epoch = 0
now_epoch = 0

loss_func = nn.CrossEntropyLoss()

def save_checkpoint(path):
	state = {'net':net.state_dict(),'optimizer':optimizer.state_dict(),'epoch':now_epoch}
	torch.save(state, path)

def load_checkpoint(path):
	checkpoint = torch.load(path)
	net.load_state_dict(checkpoint['net'])
	optimizer.load_state_dict(checkpoint['optimizer'])
	epoch = checkpoint['epoch'] + 1


def train(num_epochs):
	widgets = opt.widgets

	for e in range(epoch,num_epochs):
		now_epoch = e
		epoch_loss = 0
		pbar = progressbar.ProgressBar(widgets = widgets, maxval = len(trn)).start()
		net.train()
		for step, tree in enumerate(trn):
			net.zero_grad()
			optimizer.zero_grad()
			predictions, loss = net.getLoss(tree)
			epoch_loss += loss.item()
			loss.backward()
			clip_grad_norm_(net.parameters(), 5)
			optimizer.step()
			pbar.update(step)
		pbar.finish()

		net.eval()
		train_correctRoot,train_correctAll = net.evaluate(trn)
		dev_correctRoot, dev_correctAll = net.evaluate(dev)


		print("epoch ",e,":")
		print("train all   acc:",train_correctAll)
		print("train root  acc:",train_correctRoot)
		print("- - - - - - - - - - - - - - - - - -")
		print("dev all   acc:",dev_correctAll)
		print("dev root  acc:",dev_correctRoot)

		if opt.save_cp != 0 and (e+1) % opt.save_cp == 0:
			save_checkpoint(str(e+1) + ".pickle")


if __name__ == '__main__':
	train(opt.num_epochs)
	save_checkpoint("5_26.pickle")
	# load_checkpoint("5_26.pickle")
	# print(epoch)
	# train(20)