from parameters import *
from dataloader import vocab_size
# from torchsummary import summary
class Net(nn.Module):
	def __init__(self, vocab_size, embed_size, seq_len, labels, weight, **kwargs):
		super(Net, self).__init__(**kwargs)
		self.labels = labels
		if opt.use_embed:
			self.embedding = nn.Embedding.from_pretrained(weight)
			self.embedding.weight.requires_grad = False
		else:
			self.embedding = nn.Embedding(vocab_size, embed_size)
			self.embedding.weight.requires_grad = True
		# self.conv1 = nn.Conv2d(1, 1, (3, embed_size))
		# self.conv2 = nn.Conv2d(1, 1, (4, embed_size))
		# self.conv3 = nn.Conv2d(1, 1, (5, embed_size))
		# self.conv4 = nn.Conv2d(1, 1, (6, embed_size))
		# self.conv5 = nn.Conv2d(1, 1, (7, embed_size))
		# self.conv6 = nn.Conv2d(1, 1, (8, embed_size))
		# self.conv7 = nn.Conv2d(in_channels, out_channels, kernel_size)
		kss = [3,4,5,6,7,8,9,10,11,12]
		self.conv = nn.ModuleList([nn.Conv2d(1,1,(ks, embed_size)) for ks in kss])
		self.pool = nn.ModuleList([nn.MaxPool2d((seq_len - ks + 1,1)) for ks in kss])
		# self.pool1 = nn.MaxPool2d((seq_len - 3 + 1, 1))
		# self.pool2 = nn.MaxPool2d((seq_len - 4 + 1, 1))
		# self.pool3 = nn.MaxPool2d((seq_len - 5 + 1, 1))
		self.linear = nn.Linear(len(kss), labels)

	def forward(self, inputs):
		inputs = self.embedding(inputs).view(inputs.shape[0], 1, inputs.shape[1], -1)
		# x1 = F.relu(self.conv1(inputs))
		# x2 = F.relu(self.conv2(inputs))
		# x3 = F.relu(self.conv3(inputs))
		xs = [F.relu(self.conv[i](inputs)) for i in range(len(self.conv))]

		# print(xs[0].shape)

		# x1 = self.pool1(x1)
		# x2 = self.pool2(x2)
		# x3 = self.pool3(x3)
		xs = [self.pool[i](xs[i]) for i in range(len(self.conv))]

		# print(xs[0].shape)
		x = torch.cat([x for x in xs], -1)
		x = x.view(inputs.shape[0], 1, -1)

		x = self.linear(x)
		x = x.view(-1, self.labels)

		return(x)

if __name__ == '__main__':
	weight = torch.zeros(vocab_size+1, opt.embed_size)
	net = Net(vocab_size + 1, 100, opt.max_len, 5,weight)
	# input_tensor = torch.ones(5,opt.max_len).long()
	# output_tensor = net(input_tensor)
	# print(output_tensor.shape)

	# summary(net, (5,opt.max_len))
	print(net)
	params = net.named_parameters()
	for name,param in params:
		print(name,":",param.shape)

