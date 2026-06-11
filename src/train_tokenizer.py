from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

tokenizer = Tokenizer(
	BPE(unk_token='[UNK]')
)

tokenizer.pre_tokenizer = Whitespace()

trainer = BpeTrainer(
	vocab_size=2000,
	special_tokens=[
		'[PAD]',
		'[UNK]',
		'[BOS]',
		'[EOS]'
	]
)

tokenizer.train(
	['data/data.txt'],
	trainer
)

tokenizer.save('tokenizer.json')

print('Tokenizer saved!')