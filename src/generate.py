import os
import torch

from src.dataset import CharDataset
from src.model import TinyLM, AttentionLM
from src.tokenizer import BPETokenizer

tokenizer = BPETokenizer()

vocab_size = tokenizer.vocab_size

# Hyperparameters
context_length = 64
# batch_size = 32 # 16 for checkpoint2.pt
batch_size = 16# 16 for checkpoint2.pt
d_model = 64
learning_rate = 3e-4
# n_heads = 1 # 2 for checkpoint2.pt
# n_layers = 1 # 2 for checkpoint2.pt
n_heads = 2 # 2 for checkpoint2.pt
n_layers = 2 # 2 for checkpoint2.pt
epochs = 10

model = AttentionLM(
	vocab_size=vocab_size,
	d_model=d_model,
	context_length=context_length,
	num_heads=n_heads,
	n_layers=n_layers
)

model.load_state_dict(
	# torch.load('checkpoints/checkpoint1.pt')
	# torch.load('checkpoints/checkpoint2.pt')
	torch.load('checkpoints/checkpoint_temp_data2.pt')
)

model.eval()

@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature, context_length):
	for _ in range(max_new_tokens):
		# Contextual Cropping
		idx_cond = idx[:, -context_length:]

		logits = model(idx_cond)

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

prompt = 'hello'
# prompt = 'Hi my name is Sang Yeop'

start_tokens = tokenizer.encode(prompt)

start = torch.tensor(
	[start_tokens],
	dtype=torch.long
)

tokens = generate(
	model,
	start,
	max_new_tokens=50,
	temperature=0.3,
	context_length=context_length
)

decoded = tokenizer.decode(tokens[0].tolist())

print(decoded)