import os
import torch
from torch.utils.data import DataLoader

from src.dataset import CharDataset
from src.model import TinyLM, AttentionLM

# Used for simple token strategy
# with open('data/data.txt', 'r') as f:
# 	text = f.read()

# dataset = CharDataset(text, context_length=8)

# loader = DataLoader(
# 	dataset,
# 	batch_size=32,
# 	shuffle=True
# )

# # Model has 64 features
# # model = TinyLM(dataset.vocab_size, d_model=64)
# model = AttentionLM(vocab_size=dataset.vocab_size, d_model=64, n_layers=4)
# vocab_size = dataset.vocab_size

# optimizer = torch.optim.Adam(
# 	model.parameters(),
# 	lr=1e-3
# )