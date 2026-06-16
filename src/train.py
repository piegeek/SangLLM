import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.dataset import TextDataset
from src.tokenizer import GPTTokenizer, BPETokenizer
from src.model import AttentionLM

# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Hyperparameters
context_length = 256
batch_size = 64
d_model = 384
learning_rate = 3e-4
n_heads = 6
n_layers = 6
epochs = 10

# Dataset
dataset = TextDataset(
	'data/data.txt',
	context_length
)

loader = DataLoader(
	dataset,
	batch_size=batch_size,
	shuffle=True
)

# Tokenizer
# For full GPT params support
# tokenizer = GPTTokenizer()

# For smaller vocab_size support
tokenizer = BPETokenizer()

# Model Define (B, T, C(either vocab_size or d_model)), n_heads, n_layers
model = AttentionLM(
	vocab_size=tokenizer.vocab_size,
	d_model=d_model,
	context_length=context_length, 
	num_heads=n_heads,
	n_layers=n_layers
)

model = model.to(device)

optimizer = torch.optim.AdamW(
	model.parameters(),
	lr=learning_rate
)

# Training Loop
for i in range(epochs):
	for step, (x, y) in enumerate(loader):

		x, y = x.to(device), y.to(device)

		logits = model(x)

		B, T, C = logits.shape

		loss = F.cross_entropy(
			logits.view(B*T, C),
			y.view(B*T)
		)

		optimizer.zero_grad()

		loss.backward()

		optimizer.step()

		if step % 100 == 0:
			print(f'epoch={i} step={step} loss={loss.item():.4f}')

# Save checkpoint
torch.save(
	model.state_dict(),
	'checkpoints/checkpoint4.pt'
)

# Simple token strategy training code
# # Batch size: # of training examples, Context length = # of tokens for each example
# # Loads in batches
# # x, y has dimension (B x T)
# for x, y in loader:
# 	print(x, y)

# 	# Outputs (B, T, vocab_size)
# 	logits = model(x)

# 	# Training loss vs Validation loss
# 	loss = F.cross_entropy(
# 		logits.view(-1, vocab_size),
# 		y.view(-1)
# 	)

# 	optimizer.zero_grad()

# 	loss.backward()

# 	optimizer.step()

# 	print(loss)

# torch.save(model.state_dict(), 'model.pt')