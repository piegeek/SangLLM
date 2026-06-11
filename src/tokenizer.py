import tiktoken

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

if __name__ == '__main__':
	tok = GPTTokenizer()

	ids = tok.encode('hello world')

	print(ids)

	print(tok.decode(ids))