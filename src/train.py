import os
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.dataset import TextDataset, BinDataset
from src.tokenizer import GPTTokenizer, BPETokenizer
from src.model import AttentionLM

# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Max Steps
max_steps = 100000

# Hyperparameters
context_length = 256
batch_size = 64
d_model = 384
learning_rate = 3e-4
n_heads = 6
n_layers = 6
epochs = 10

# --- LR schedule settings (warmup + cosine decay) ---
warmup_steps = 1000
min_lr = learning_rate * 0.1
grad_clip = 1.0

train_ds = BinDataset('data/train.bin', context_length)
val_ds = BinDataset('data/val.bin', context_length)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

# For smaller vocab_size support; We need tokenizer to get vocab size and in generate.py to encode, decode
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

# --- 1 + 2: learning-rate warmup, then cosine decay ---
def get_lr(step):
	if step < warmup_steps:
		return learning_rate * (step + 1) / warmup_steps

	if step >= max_steps:
		return min_lr

	# Between warmup_steps and max_steps
	# Cosine decay
	progress = (step - warmup_steps) / (max_steps - warmup_steps)
	coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
	
	return min_lr + coeff * (learning_rate - min_lr)

# Evaluate val_dataset
@torch.no_grad()
def evaluate(max_batches=20):
	model.eval()
	losses = []
	for step, (x, y) in enumerate(val_loader):
		if step >= max_batches:
			break
		x, y = x.to(device), y.to(device)
		logits = model(x)
		B, T, C = logits.shape
		loss = F.cross_entropy(logits.view(B*T, C), y.view(B*T))
		losses.append(loss.item())
	model.train()
	return sum(losses) / len(losses)

global_step = 0
# Training Loop
for i in range(epochs):
	for step, (x, y) in enumerate(train_loader):
		# Set this step's learning rate (warmup + cosine) on the optimizer
		lr = get_lr(global_step)
		for param_group in optimizer.param_groups:
			param_group['lr'] = lr

		x, y = x.to(device), y.to(device)

		logits = model(x)

		B, T, C = logits.shape

		loss = F.cross_entropy(
			logits.view(B*T, C),
			y.view(B*T)
		)

		optimizer.zero_grad()

		loss.backward()
		
		# --- 3: gradient clipping (cap the global grad norm before stepping) ---
		torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

		optimizer.step()

		if global_step % 100 == 0:
			val_loss = evaluate()
			print(f'epoch={i} step={global_step} lr={lr:.2e} train_loss={loss.item():.4f}, val_loss={val_loss:.4f}')

		global_step += 1

		if global_step >= max_steps:
			break
	
	if global_step >= max_steps:
		break

# Save checkpoint
torch.save(
	model.state_dict(),
	'checkpoints/checkpoint5.pt'
)
