import torch
import torch.nn as nn
import torch.nn.functional as F

class Head(nn.Module):
	def __init__(self, d_model, head_size):
		super().__init__()
		
		# (B, T, d_model) -> (B, T, head_size)
		self.query = nn.Linear(
			d_model,
			head_size,
			bias=False
		)

		self.key = nn.Linear(
			d_model,
			head_size,
			bias=False
		)

		self.value = nn.Linear(
			d_model,
			head_size,
			bias=False
		)

		# Causal Mask
		self.register_buffer(
			'tril',
			torch.tril(
				torch.ones(1024, 1024)
			)
		)

	def forward(self, x):
		q = self.query(x)
		k = self.key(x)
		v = self.value(x)

		B, T, C = x.shape

		# (B, T, 16) * (B, 16 ,T) -> (B, T, T) <---- Attention matrix
		weights = q @ k.transpose(-2, -1)

		# Scaling
		weights = weights / (k.shape[-1] ** 0.5)

		weights = weights.masked_fill(
			self.tril[:T, :T] == 0,
			float('-inf')
		)

		# Softmax
		weights = F.softmax(
			weights,
			dim=-1
		)

		# Weighted sum of values (B, T, T) * (B, T, 16)
		out = weights @ v

		return out

class MultiHeadAttention(nn.Module):
	def __init__(self, d_model, num_heads):
		super().__init__()

		# When num_heads = 1 (Single Head), head_size = d_model
		head_size = d_model // num_heads

		self.heads = nn.ModuleList(
			[Head(d_model, head_size) for _ in range(num_heads)]
		)

		self.proj = nn.Linear(d_model, d_model)

		# Transformers almost always use dropout.
		self.dropout = nn.Dropout(0.2)

	def forward(self, x):
		out = torch.cat(
			[h(x) for h in self.heads],
			dim=-1
		)

		out = self.proj(out)
		
		# Transformers almost always use dropout.
		out = self.dropout(out)

		return out

if __name__ == '__main__':
	head = Head(
		d_model=64,
		head_size=16
	)

	x = torch.randn(
		32,
		8,
		64
	)

	out = head(x)

	print(out.shape)