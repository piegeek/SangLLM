import tiktoken
from tokenizers import Tokenizer

class GPTTokenizer:
	def __init__(self):
		self.enc = tiktoken.get_encoding(
			'cl100k_base'
		)

	@property
	def vocab_size(self):
		return self.enc.n_vocab

	def encode(self, text):
		return self.enc.encode(text)

	def decode(self, tokens):
		return self.enc.decode(tokens)

class BPETokenizer:
	def __init__(self,path='tokenizer.json'):
		self.tokenizer = Tokenizer.from_file(path)

	@property
	def vocab_size(self):
		return self.tokenizer.get_vocab_size()

	def encode(self, text):
		return self.tokenizer.encode(text).ids

	def decode(self, ids):
		return self.tokenizer.decode(ids)

if __name__ == '__main__':
	tok = GPTTokenizer()

	ids = tok.encode('hello world')

	print(ids)

	print(tok.decode(ids))