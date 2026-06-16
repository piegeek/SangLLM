import torch 

from transformers import (
	GPT2LMHeadModel,
	GPT2Tokenizer
)

tokenizer = GPT2Tokenizer.from_pretrained('checkpoints/gpt2_sentiment')

model = GPT2LMHeadModel.from_pretrained('checkpoints/gpt2_sentiment')

model.eval()

prompt = '''
Tweet: I absolutely love this movie.
Sentiment:
'''

inputs = tokenizer(
	prompt,
	return_tensors='pt'
)

with torch.no_grad():
	output = model.generate(
		**inputs,
		max_new_tokens=5,
		do_sample=False
	)

print(
	tokenizer.decode(
		output[0],
		skip_special_tokens=True
	)
)