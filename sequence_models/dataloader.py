# import modules
from torchtext import data, vocab
import pandas as pd


def Tokenize():
	'''
	Method of tokenization
	# TO DO: prepare to make adjustments incase we have different fields
	that require different tokenization
	'''
	tokenize = lambda x: list(x)
	return tokenize


def BuildDataset(path, alphabet, field_params):
	'''
	Converts CSV to torchtext Dataset and sets tokens
	
	Parameters:
	-----------
	path : str 
		location of train, test and validation data
	
	alphabet : str
		string of ordered alphabet for one-hot representation 

	field_params : dict of dict
		dictionary with key as field name and value as dictionary of
		torchtext Field() object parameters 
		
	Returns:
	--------
	train : torchtext Dataset
		train dataset

	test : torchtext Dataset
		test dataset

	val : torchtext Dataset
		validation dataset
	
	'''

	fields = [(k, data.Field(**v)) for k,v in field_params.items()]

	train, val, test = data.TabularDataset.splits(
		path=path, train='train.csv',
		validation='valid.csv', test='test.csv', format='csv', skip_header=True,
		fields=fields)
	
	for field in fields:
		if field[0] == 'sequences':
			
			# initiate vocab
			field[1].build_vocab(train)
	
			# order vocab according to alphabet order
			field[1].vocab.stoi = {a:i for i, a in enumerate(alphabet)}

	return train, test, val


# set up parameters for batch size selection 
global APPROX_TOKENS # approx. number of tokens per batch
global LONGEST_EXAMPLE # track the longest example
global MAX_BATCH_SIZE # maximum batch size 

def batch_size_fn(new, count, sofar):
	'''
	Function to load into dataloader to choose batch sizes with approx.
	same number of tokens 

	Parameters:
	-----------
	new : torchtext Example

	count : current count of examples in the batch

	sofar : current effective batch size

	
	Returns:
	--------
	count : int, global variable
		Current count of samples in batch, tells dataloader it can
		continue to add examples to this batch

	max_batch : int, global variable
		Tells dataloader to stop adding new examples and yield batch
	'''
	global APPROX_TOKENS, MAX_BATCH_SIZE, LONGEST_EXAMPLE
	if count == 1:
		LONGEST_EXAMPLE = len(new.sequences)
	LONGEST_EXAMPLE = max(LONGEST_EXAMPLE, len(new.sequences))
	if LONGEST_EXAMPLE*count <= APPROX_TOKENS:
		return count
	else:
		return MAX_BATCH_SIZE


def BuildDataLoader(train, test, val):
	'''
	Build iterator for train, test and val datasets that produces
	batches with similar length sequences
	
	Parameters:
	-----------
	train : dataset
	
	test : dataset
	
	val : dataset
	
	batch_sizes : tuple (x,y,z)
		Tuple containes batch size for train, test and val loader 
		
	
	Returns:
	--------
	iter(train_dl) : iterator
		train set dataloader 

	iter(test_dl) : iterator
		test set dataloader 

	iter(val_dl) : iterators
		validation set dataloader
	
	'''
	
	global MAX_BATCH_SIZE
	train_dl, test_dl, val_dl = data.BucketIterator.splits(
		(train, test, val),
		batch_sizes=(MAX_BATCH_SIZE, MAX_BATCH_SIZE, MAX_BATCH_SIZE),
		batch_size_fn=batch_size_fn,
		sort_key=lambda x: len(x.sequences),
		sort_within_batch=True,
		repeat=False
	)
	
	return iter(train_dl), iter(test_dl), iter(val_dl)


# Global variable assignment 
# APPROX_TOKENS = 1000
# LONGEST_EXAMPLE = 0
# AX_BATCH_SIZE = 100
