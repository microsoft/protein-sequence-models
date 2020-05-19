# import modules
from torchtext import data, vocab
import pandas as pd



def tokenize(tokenize_method=None): 
	'''
	Method of tokenization
	# TO DO: prepare to make adjustments incase we have different fields
	that require different tokenization
	'''

	if tokenize_method is not None:
		return tokenize_method
	else:
		tokenize_method = lambda x: list(x)
		return tokenize_method



class BuildDataset:

	def __init__(self,):
		self.sequence_stoi = None


	def generate_dataset(self, path, alphabet, field_params):

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
				self.sequence_stoi = field[1].vocab.stoi 

		return train, test, val



class BuildDataLoader:

	def __init__(self, approx_tokens, max_batch_size):

		self.approx_tokens = approx_tokens
		self.max_batch_size = max_batch_size
		self.longest_example = 0


	def batch_size_fn(self, new, count, sofar):
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
		if count == 1:
			self.longest_example = len(new.sequences)
		self.longest_example = max(self.longest_example, len(new.sequences))
		if self.longest_example*count <= self.approx_tokens:
			return count
		else:
			return self.max_batch_size

	def generate_loader(self, train, test, val):
		'''
		Build iterator for train, test and val datasets that produces
		batches with similar length sequences
		
		Parameters:
		-----------
		train : dataset
		
		test : dataset
		
		val : dataset
			
		
		Returns:
		--------
		iter(train_dl) : iterator
			train set dataloader 

		iter(test_dl) : iterator
			test set dataloader 

		iter(val_dl) : iterators
			validation set dataloader
		
		'''
		
		train_dl, test_dl, val_dl = data.BucketIterator.splits(
			(train, test, val),
			batch_sizes=(self.max_batch_size, self.max_batch_size, self.max_batch_size),
			batch_size_fn=self.batch_size_fn,
			sort_key=lambda x: len(x.sequences),
			sort_within_batch=True,
			repeat=False
		)
		
		return iter(train_dl), iter(test_dl), iter(val_dl)
