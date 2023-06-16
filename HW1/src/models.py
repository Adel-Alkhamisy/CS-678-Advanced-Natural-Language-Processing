# models.py
from torch.autograd._functions import tensor
import torch.nn.functional as F

from sentiment_data import *
from evaluator import *

from collections import Counter
import os
import numpy as np
import torch
from torch import nn, optim


######################################
# IMPLEMENT THE SENTIMENT CLASSIFIER #
######################################

class FeedForwardNeuralNetClassifier(nn.Module):
    """
    The Feed-Forward Neural Net sentiment classifier.
    """
    def __init__(self, args, vocab, n_classes, vocab_size, emb_dim, n_hidden_units):
        """
        In the __init__ function, you will define modules in FFNN.
        :param n_classes: number of classes in this classification problem
        :param vocab_size: size of vocabulary
        :param emb_dim: dimension of the embedding vectors
        :param n_hidden_units: dimension of the hidden units
        """
        super(FeedForwardNeuralNetClassifier, self).__init__()
        self.n_classes = n_classes
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.n_hidden_units = n_hidden_units
       
        # TODO: create a randomly initialized embedding matrix, and set padding_idx as 0
        # PAD's embedding will not be trained and by default is initialized as zero
        if args.glove_path is not None:
            # load the embedding
            embeddings = dict()
            file = open(args.glove_path)
            for line in file:
                values = line.split()
                word = values[0]
                c = np.asarray(values[1:], dtype='float32')
                embeddings[word] = c
            # create a weight matrix for words in vocab
            embedding_matrix = np.zeros((len(vocab), 300))
            for i in range(0,len(vocab)):
                vector = embeddings.get(vocab[i])
                if vector is not None:
                    embedding_matrix[i] = vector

            #create representation of unknown words as the average of all the pretrianed words vectors which is very rare to occur
            with open(args.glove_path, 'r') as f:
                for i, line in enumerate(f):
                    pass
            vec_numbers = i + 1
            hidden_dim = len(line.split(' ')) - 1
            v = np.zeros((vec_numbers, hidden_dim), dtype=np.float32)
            with open(args.glove_path, 'r') as f:
                for i, line in enumerate(f):
                    v[i] = np.array([float(x) for x in line.split(' ')[1:]], dtype=np.float32)
            average_vector_value = np.mean(v, axis=0)
            # set embedding_matrix[1] to the representation of UKN words
            embedding_matrix[1] = average_vector_value
            # note in the above code we let zeros to be the representation of padding

            self.word_embeddings = nn.Embedding(self.vocab_size, self.emb_dim, padding_idx=0)
            self.word_embeddings.weight.data.copy_(torch.from_numpy(embedding_matrix))
            self.word_embeddings.weight.requires_grad = False
        else:
            self.word_embeddings = nn.Embedding(self.vocab_size, self.emb_dim, padding_idx=0)
            self.word_embeddings.weight.data.uniform_(-1, 1)
        # TODO: implement the FFNN architecture
        # when you build the FFNN model, you will need specify the embedding size using self.emb_dim, the hidden size using self.n_hidden_units,
        # and the output class size using self.n_classes        

        self.input_Layer = nn.Linear(self.emb_dim, self.n_hidden_units)
        self.Relu_activation = nn.ReLU()
        self.h_out = nn.Linear(self.n_hidden_units, self.n_classes)

    def forward(self, batch_inputs: torch.Tensor, batch_lengths: torch.Tensor) -> torch.Tensor:
        """
        The forward function, which defines how FFNN should work when given a batch of inputs and their actual sent lengths (i.e., before PAD)
        :param batch_inputs: a torch.Tensor object of size (n_examples, max_sent_length_in_this_batch), which is the *indexed* inputs
        :param batch_lengths: a torch.Tensor object of size (n_examples), which describes the actual sentence length of each example (i.e., before PAD)
        :return the logits outputs of FFNN (i.e., the unnormalized hidden units before softmax)
        """
        # TODO: implement the forward function, which returns the

        x = self.word_embeddings(batch_inputs)
        av = torch.mean(x, axis=1)

        # Linear function Wh · av + bh
        output = self.input_Layer(av)
        # ReLU Non-linear function  h = ReLU(Wh · av + bh),
        output = self.Relu_activation(output)
        # Linear function h_out = W_out · h + b_out
        output = self.h_out(output)

        return output
        raise Exception("Not Implemented!")
        
    
    def batch_predict(self, batch_inputs: torch.Tensor, batch_lengths: torch.Tensor) -> List[int]:
        """
        Make predictions for a batch of inputs. This function may directly invoke `forward` (which passes the input through FFNN and returns the output logits)

        :param batch_inputs: a torch.Tensor object of size (n_examples, max_sent_length_in_this_batch), which is the *indexed* inputs
        :param batch_lengths: a torch.Tensor object of size (n_examples), which describes the actual sentence length of each example (i.e., before PAD)
        :return: a list of predicted classes for this batch of data, either 0 for negative class or 1 for positive class
        """
        # TODO: implement the prediction function, which could reuse the forward function 
        # but should return a list of predicted labels
        predictions = []
        out = self.forward(batch_inputs, batch_lengths)

        pred = F.softmax(out, dim=1)

        for i in pred:
            if i[0]>i[1]:
                predictions.append(0)
            else:
                predictions.append(1)
        return predictions
        raise Exception("Not Implemented!")


##################################
# IMPLEMENT THE TRAINING METHODS #
##################################

def train_feedforward_neural_net(
    args,
    train_exs: List[SentimentExample], 
    dev_exs: List[SentimentExample]) -> FeedForwardNeuralNetClassifier:
    """
    Main entry point for your modifications. Trains and returns a FFNN model (whose architecture is configured based on args)

    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :param dev_exs: dev set, List of SentimentExample objects. You can use this for validation throughout the training
    process, but you should *not* directly train on this data.
    :return: trained SentimentClassifier model, of whichever type is specified
    """

    # TODO: read in all training examples and create a vocabulary (a List-type object called `vocab`)
    vocab = [] # replace None with the correct implementation
    corpus=[]

    for i in train_exs:
        for j in i.words:
            corpus.append(j)
    # create a dictionary and count the occurrences of unique words
    dic_word_counter = Counter(corpus)
    for w in corpus:
        dic_word_counter.update(w)
    corpus = {x: y for x, y in sorted(dic_word_counter.items(), key=lambda item: item[1])}

    # store dictionary's keys (unique words) in vocab to form vocabulary list
    vocab = list(corpus.keys())
    # add PAD and UNK as the first two tokens
    # DO NOT CHANGE, PAD must go first and UNK next (as their indices have been hard-coded in several places)
    vocab = ["PAD", "UNK"] + vocab
    print("Vocab size:", len(vocab))
    # write vocab to an external file, so the vocab can be reloaded to index the test set
    with open("data/vocab.txt", "w") as f:
        for word in vocab:
            f.write(word + "\n")

    # indexing the training/dev examples
    indexing_sentiment_examples(train_exs, vocabulary=vocab, UNK_idx=1)
    indexing_sentiment_examples(dev_exs, vocabulary=vocab, UNK_idx=1)

    # TODO: create the FFNN classifier
    model = FeedForwardNeuralNetClassifier(args, vocab, n_classes=2, vocab_size=len(vocab), emb_dim=args.emb_dim, n_hidden_units=args.n_hidden_units)
    loss_function = nn.CrossEntropyLoss()
    # TODO: define an Adam optimizer, using default config
    if args.glove_path is None:
        optimizer = optim.Adam(model.parameters())
    # optimizer = optim.SGD(model.parameters(), lr=0.00001)
    else:
        # pass para that has param.requires_grad == True mean that the model will not update word embedding in case of pretrained GLOVE
        optimizer = optim.Adam([param for param in model.parameters() if param.requires_grad == True])
    
    # create a batch iterator for the training data
    batch_iterator = SentimentExampleBatchIterator(train_exs, batch_size=args.batch_size, PAD_idx=0, shuffle=True)

    # training
    best_epoch = -1
    best_acc = -1
    for epoch in range(args.n_epochs):
        print("Epoch %i" % epoch)

        batch_iterator.refresh() # initiate a new iterator for this epoch

        model.train() # turn on the "training mode"
        batch_loss = 0.0
        batch_example_count = 0
        batch_data = batch_iterator.get_next_batch()
        while batch_data is not None:
            batch_inputs, batch_lengths, batch_labels = batch_data
            # TODO: clean up the gradients for this batch
            optimizer.zero_grad()

            # TODO: call the model to get the logits
            logits = model(batch_inputs, batch_lengths)

            # TODO: calculate the loss (let's name it `loss`, so the follow-up lines could collect the stats)
            loss = loss_function(logits, batch_labels)

            # record the loss and number of examples, so we could report some stats
            batch_example_count += len(batch_labels)
            batch_loss += loss.item() * len(batch_labels)

            # TODO: backpropagation (backward and step)
            loss.backward()

            # Updating parameters
            optimizer.step()

            # get another batch
            batch_data = batch_iterator.get_next_batch()

        print("Avg loss: %.5f" % (batch_loss / batch_example_count))

        # evaluate on dev set
        model.eval() # turn on the "evaluation mode"
        acc, _, _, _ = evaluate(model, dev_exs, return_metrics=True)
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch
            print("Secure a new best accuracy %.3f in epoch %d!" % (best_acc, best_epoch))
            
            # save the current best model parameters
            print("Save the best model checkpoint as `best_model.ckpt`!")
            torch.save(model.state_dict(), "best_model.ckpt")
        print("-" * 10)

    # load back the best checkpoint on dev set
    model.load_state_dict(torch.load("best_model.ckpt"))
    
    model.eval() # switch to the evaluation mode
    return model
