from parameters import *

class Net(nn.Module):
	def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
				 bidirectional, weight, labels, **kwargs):
		super(Net, self).__init__(**kwargs)
		self.num_hiddens = num_hiddens
		self.num_layers = num_layers
		self.bidirectional = bidirectional
		if opt.use_embed:
			self.embedding = nn.Embedding.from_pretrained(weight)
			self.embedding.weight.requires_grad = False
		else:
			self.embedding = nn.Embedding(vocab_size, embed_size)
			self.embedding.weight.requires_grad = True
		self.encoder = nn.LSTM(input_size=embed_size, hidden_size=self.num_hiddens,
							   num_layers=num_layers, bidirectional=self.bidirectional,
							   dropout=0)
		if self.bidirectional:
			self.decoder = nn.Linear(num_hiddens * 2, labels)
		else:
			self.decoder = nn.Linear(num_hiddens, labels)

	def forward(self, inputs):
		embeddings = self.embedding(inputs)
		# print(embeddings.shape)
		states, hidden = self.encoder(embeddings.permute([1, 0, 2]))
		# print(states.shape)
		# encoding = torch.cat([states[0], states[-1]], dim=1) # 使用初始状态和最后状态的拼接
		
		outputs = self.decoder(states[-1])
		return outputs


if __name__ == '__main__':
	net = Net(5, 100, 100, 2, True, None, 5)
	input_tensor = torch.ones(5,48).long()
	output_tensor = net(input_tensor)
	print(output_tensor.shape)