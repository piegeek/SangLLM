import torch
import torch.nn as nn

from src.attention import Head, MultiHeadAttention

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
	def __init__(self, vocab_size, d_model, context_length=8, n_layers=2):
		super().__init__()

		self.token_embedding = nn.Embedding(vocab_size, d_model)
		self.position_embedding = nn.Embedding(context_length, d_model)

		# Single-headed attention
		# self.attention = Head(
		# 	d_model=d_model,
		# 	head_size=d_model
		# )

		# Multi-headed attention
		self.attention = MultiHeadAttention(
			d_model=d_model,
			num_heads=4
		)

		# Blocks
		self.blocks = nn.Sequential(
			*[TransformerBlock(d_model, self.attention) for _ in range(n_layers)]
		)

		# Final layer LayerNorm
		self.ln_f = nn.LayerNorm(d_model)

		self.lm_head = nn.Linear(
			d_model,
			vocab_size
		)

	def forward(self, idx):
		B, T = idx.shape

		tok_emb = self.token_embedding(idx)

		pos_emb = self.position_embedding(
			torch.arange(T, device=idx.device)
		)

		x = tok_emb + pos_emb

		x = self.blocks(x)

		x = self.ln_f(x)

		logits = self.lm_head(x)

		return logits

class FeedForward(nn.Module):
	def __init__(self, d_model):
		super().__init__()

		self.net = nn.Sequential(
			nn.Linear(d_model, 4 * d_model),
			nn.GELU(),
			nn.Linear(4 * d_model, d_model)
		)

		self.dropout = nn.Dropout(0.2)

	def forward(self, x):
		out = self.net(x)
		
		# Transformers almost always use dropout.
		out = self.dropout(out)
	
		return out

class TransformerBlock(nn.Module):
	def __init__(self, d_model, attention_module):
		super().__init__()

		self.attn = attention_module
		self.ffn = FeedForward(d_model)

		self.ln1 = nn.LayerNorm(d_model)
		self.ln2 = nn.LayerNorm(d_model)

	def forward(self, x):
		# Attention sublayer
		x = x + self.attn(self.ln1(x))

		# Feedforward sublayer
		x = x + self.ffn(self.ln2(x))

		return x

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