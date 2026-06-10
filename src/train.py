import os
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from src import dataset, loader, model, vocab_size, optimizer

# Batch size: # of training examples, Context length = # of tokens for each example
# Loads in batches
# x, y has dimension (B x T)
for x, y in loader:
	print(x, y)

	# Outputs (B, T, vocab_size)
	logits = model(x)

	# Training loss vs Validation loss
	loss = F.cross_entropy(
		logits.view(-1, vocab_size),
		y.view(-1)
	)

	optimizer.zero_grad()

	loss.backward()

	optimizer.step()

	print(loss)

torch.save(model.state_dict(), 'model.pt')