import torch.nn as nn

class TinyLM(nn.Module):
	def __init__(self, vocab_size, d_model):
		super().__init__()

		# token_num x features
		self.embedding = nn.Embedding(
			vocab_size,
			d_model
		)

		# Used for the final step
		self.head = nn.Linear(
			d_model,
			vocab_size
		)

	def forward(self, x):
		x = self.embedding(x)

		logits = self.head(x)

		return logits

class BigramLM(nn.Module):
	def __init__(self, vocab_size):
		super().__init__()

		# Current x Next
		self.table = nn.Embedding(
			vocab_size,
			vocab_size
		)

	def forward(self, idx):
		logits = self.table(idx)

		return logits