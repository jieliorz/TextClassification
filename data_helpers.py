import numpy as np
import re
from sklearn.model_selection import train_test_split
import random

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r", encoding='utf-8').readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r", encoding='utf-8').readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = positive_labels + negative_labels
    return [x_text, y]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data_size = len(data)
    num_batches_per_epoch = int((data_size-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            random.shuffle(data)
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield data[start_index:end_index]

def preprocess(positive_data_file, negative_data_file):
    print("Loading data...")
    x_text, y = load_data_and_labels(positive_data_file, negative_data_file)

    # Build vocabulary
    vocabulary = {word for x in x_text for word in x.split(" ")}
    vocab = {}
    for word in vocabulary:
        vocab[word]=len(vocab)
    vocab[''] = len(vocab)
    vocab['<pad>'] = len(vocab)
    vocab['<unk>'] = len(vocab)

    # vocab_size=len(vocab)

    x_text=[x.split(" ") for x in x_text]

    document_length = [len(x) for x in x_text]
    max_document_length=max(document_length)
    [x.extend(['<pad>']*(max_document_length-len(x))) for x in x_text]

    x_embed=[[vocab[word] for word in x] for x in x_text]

    dataset=list(zip(x_embed,y,document_length))
    train_set, test_set =train_test_split(dataset)
    # (x_embed,y,real_length)
    return vocab,max_document_length,train_set, test_set



if __name__ == '__main__':
    positive_data_file='rt-polaritydata/rt-polarity.neg'
    negative_data_file='rt-polaritydata/rt-polarity.pos'
    preprocess(positive_data_file, negative_data_file)
