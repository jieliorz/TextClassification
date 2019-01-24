from cnnModel import TextCNN
from data_helpers import *
from rnnModel import TextRnn


if __name__ == '__main__':

	positive_data_file='rt-polaritydata/rt-polarity.pos'
	negative_data_file='rt-polaritydata/rt-polarity.neg'
	vocab,max_document_length,train_set,test_set = preprocess(positive_data_file,negative_data_file)
	vocab_size=len(vocab)
		

	# model=TextCNN(embedding_size=64,
	# 				vocab_size=vocab_size,
	# 				input_size=max_document_length,
	# 				num_classes=2,
	# 				filter_sizes=[3,5,7],
	# 				droupout=0.5,
	# 				batch_size=100,
	# 				learning_rate=0.02,
	# 				num_filters=1,
	# 				num_epochs=10,istrain=True)


	model=TextRnn(embedding_size=64,
					vocab_size=vocab_size,
					input_size=max_document_length,
					num_classes=2,
					droupout=0.5,
					batch_size=100,
					learning_rate=0.02,
					hidden_size=64,
					num_epochs=10,istrain=True)
	model.train(train_set)
	model.test(test_set)

