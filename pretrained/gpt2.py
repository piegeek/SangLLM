from transformers import GPT2LMHeadModel
from transformers import GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

prompt = 'Once upon a time'

inputs = tokenizer(
	prompt,
	return_tensors='pt'
)

output = model.generate(
	**inputs,
	max_new_tokens=50
)

output_text = tokenizer.decode(
	output[0],
	skip_special_tokens=True
)

print(output_text)