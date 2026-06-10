import torch
import torch.nn as nn

from src.attention import Head

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

class AttentionLM(nn.Module):
	def __init__(self, vocab_size, d_model):
		super().__init__()

		self.embedding = nn.Embedding(
			vocab_size,
			d_model
		)

		self.attention = Head(
			d_model=d_model,
			head_size=d_model
		)

		self.lm_head = nn.Linear(
			d_model,
			vocab_size
		)

	def forward(self, idx):
		x = self.embedding(idx)

		x = self.attention(x)

		logits = self.lm_head(x)

		return logits

if __name__ == '__main__':
	model = AttentionLM(17, 64)

	# 32 x 8 matrix that holds values 0-17
	idx = torch.randint(
		0, 
		17,
		(32, 8) 
	)

	logits = model(idx)

	print(logits.shape)