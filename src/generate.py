import os
import torch

from src.dataset import CharDataset
from src.model import TinyLM, AttentionLM

with open('data/data.txt', 'r') as f:
	text = f.read()

dataset = CharDataset(text)
vocab_size = dataset.vocab_size

model = AttentionLM(
	vocab_size=vocab_size,
	d_model=64,
	n_layers=4
)

model.load_state_dict(
	torch.load('model.pt')
)

model.eval()

@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature):
	for _ in range(max_new_tokens):
		logits = model.forward(idx)

		logits = logits[:, -1, :]

		# Add temperature
		logits = logits / temperature

		probs = torch.softmax(logits, dim=-1)

		next_token = torch.multinomial(
			probs,
			num_samples=1
		)

		idx = torch.cat(
			[idx, next_token],
			dim=1
		)

	return idx

start = torch.tensor([
	[dataset.stoi['h']]
])

tokens = generate(
	model,
	start,
	max_new_tokens=1,
	temperature=0.8
)

decoded = ''.join(
	dataset.itos[i.item()] for i in tokens[0]
)

print(decoded)