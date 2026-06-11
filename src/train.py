import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.dataset import TextDataset
from src.tokenizer import GPTTokenizer
from src.model import AttentionLM

# Hyperparameters
context_length = 64
batch_size = 32
d_model = 64
learning_rate = 3e-4
n_heads = 1
n_layers = 1
epochs = 10

# Tokenizer
tokenizer = GPTTokenizer()

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

print('Dataset size:', len(dataset))
print('Num batches:', len(loader))
print(tokenizer.vocab_size)

# Model
model = AttentionLM(
	vocab_size=tokenizer.vocab_size,
	d_model=d_model,
	context_length=context_length,
	num_heads=n_heads,
	n_layers=n_layers
)

optimizer = torch.optim.AdamW(
	model.parameters(),
	lr=learning_rate
)

# Training Loop
for i in range(epochs):
	for step, (x, y) in enumerate(loader):
		
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
	'checkpoint.pt'
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