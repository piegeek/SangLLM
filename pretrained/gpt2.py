from transformers import GPT2LMHeadModel
from transformers import GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

prompt = 'Once upon a time'

inputs = tokenizer(
	prompt,
	return_tensors='pt'
)

# Greedy decoding
# output = model.generate(
# 	**inputs,
# 	max_new_tokens=50
# )

# With Sampling 
output = model.generate(
	**inputs,
	max_new_tokens=50,
	do_sample=True,
	temperature=0.8,
	top_k=50,
	# top_p=0.95
)

output_text = tokenizer.decode(
	output[0],
	skip_special_tokens=True
)

print(output_text)