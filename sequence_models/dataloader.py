# import modules
from torchtext import data, vocab
import os
import pandas as pd
import matplotlib.pyplot as plt

def BuildDataset(path, alphabet_to_token):
    '''
    Converts CSV to Pytorch Dataset and sets tokens
    
    Parameters:
    -----------
    path : str 
        location of train, test and validation data
    
    alphabet_to_token : dict
        dictionary mapping string to one-hot representation
        
    Returns:
    --------
    train, test, val : Pytorch Dataset
    
    '''
    
    tokenize = lambda x: list(x)
    SEQUENCES = data.Field(sequential=True, lower = False, tokenize=tokenize, # , 
                          pad_token='*', )

    train, val, test = data.TabularDataset.splits(
        path=path, train='train.csv',
        validation='valid.csv', test='test.csv', format = 'csv', skip_header = True,
        fields=[('sequences', SEQUENCES), ])
    
    SEQUENCES.build_vocab(train,)
    
    # order vocab according to alphabet order 
    SEQUENCES.vocab.stoi = alphabet_to_token
    
    return train, test, val
    

def BuildDataLoader(train, test, val, batch_sizes):
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
    train_dl, test_dl, val_dl : iterators
    
    '''
    train_dl, test_dl, val_dl = data.BucketIterator.splits(
        (train, test, val),
        batch_sizes = batch_sizes,
        sort_key = lambda x: len(x.sequences),
        sort_within_batch = True,
        repeat = False
    )
    
    return iter(train_dl), iter(test_dl), iter(val_dl)
    