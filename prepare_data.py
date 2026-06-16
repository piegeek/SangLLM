import numpy as np
from datasets import load_dataset
from src.tokenizer import BPETokenizer

# provides 'train' and 'validation'
ds = load_dataset('roneneldan/TinyStories')

# (Re)train your BPE on this corpus first — point train_tokenizer.py at a text
# dump of ds['train'] with vocab_size=8000, then load it here:
tok = BPETokenizer()

def to_bin(split, path):
	ids = []

	for ex in ds[split]:
		ids.extend(tok.encode(ex['text']))

	arr = np.array(ids, dtype=np.uint16)
	arr.tofile(path)
	print(split, len(arr), 'tokens ->', path)

to_bin('train', 'data/train.bin')
to_bin('validation', 'data/val.bin')