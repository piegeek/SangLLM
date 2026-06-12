from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, DataCollatorForLanguageModeling, TrainingArguments, Trainer

# TODO: Personal HF token

dataset = load_dataset('mteb/tweet_sentiment_extraction')

# Load GPT2
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained('gpt2')

# Tokenize training examples
def tokenize(example):
	text = format_example(example)

	enc = tokenizer(
		text,
		truncation=True,
		max_length=128
	)

	# Create lables
	enc['labels'] = enc['input_ids'].copy()

	return enc

def format_example(example):
	return (
		f"Tweet: {example['text']}\n"
		f"Sentiment: {example['label_text']}"
	)

tokenized = dataset.map(tokenize)

# Remove unused columns
tokenized = tokenized.remove_columns(
	[
		'text',
		'label',
		'label_text'
	]
)

# Data collator
collator = DataCollatorForLanguageModeling(
	tokenizer=tokenizer,
	mlm=False
)

# Training arguments
args = TrainingArguments(
	output_dir='./gpt2_sentiment',
	learning_rate=5e-5,
	per_device_train_batch_size=8,
	num_train_epochs=3,
	logging_steps=100,
	save_strategy='epoch'
)

# Train
trainer = Trainer(
	model=model,
	args=args,
	train_dataset=tokenized['train'],
	eval_dataset=tokenized['test'],
	data_collator=collator
)

trainer.train()

# Save
model.save_pretrained(
	'checkpoints/gpt2_sentiment'
)

tokenizer.save_pretrained(
	'checkpoints/gpt2_sentiment'
)

