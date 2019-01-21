,import tensorflow as tf
from data_helpers import *

positive_data_file='rt-polaritydata/rt-polarity.pos'
negative_data_file='rt-polaritydata/rt-polarity.neg'
vocab,max_document_length,train_set,test_set = preprocess(positive_data_file,negative_data_file)
vocab_size=len(vocab)



def get_batch(self):
	for batch in batch_iter(self.train_set, self.batch_size, self.num_epochs, shuffle=True):
		x_embed,y,real_length = batch

