import torch
from torch.utils.data import Dataset

import os

class CharDataset(Dataset):
	def __init__(self, text, context_length=8):
		chars = sorted(list(set(text)))

		self.stoi = {c : i for i, c in enumerate(chars)}
		self.itos = {i : c for i, c in enumerate(chars)}

		self.vocab_size = len(chars)

		self.tokens = [ self.stoi[c] for c in text ]

		self.context_length = context_length

	def __len__(self):
		return len(self.tokens) - self.context_length

	def __getitem__(self, idx):
		x = torch.tensor(
			self.tokens[idx:idx+self.context_length]
		)

		y = torch.tensor(
			self.tokens[idx+1:idx+self.context_length+1]
		)

		return x, y

if __name__ == '__main__':
	current_dir = os.path.dirname(os.path.abspath(__file__))

	file_path = os.path.join(current_dir, '..', 'data', 'data.txt')

	text = []
	with open(file_path, 'r') as file:
		while True:
			content = file.readline()

			if not content:
				break

			text.append(content)

	for t in text:
		dataset = CharDataset(t)
		x, y = dataset[0]

		print(x)
		print(y)