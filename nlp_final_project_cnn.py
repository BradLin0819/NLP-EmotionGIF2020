# -*- coding: utf-8 -*-

import torch
import nltk
from nltk.corpus import stopwords
from torchtext import data
from torchtext import datasets
import torch
import json
import numpy as np
from torchtools import EarlyStopping
import argparse
import re

categories = None

with open('categories.json') as f:
  categories = json.load(f)

categories = np.array(categories)

def make_one_hot(labels):
  one_hot_labels = np.zeros(len(categories))

  for label in labels:
    one_hot_labels[categories == label] = 1.0
  return one_hot_labels

def tokenize(sentence):
    return [token for token in tokenizer.tokenize(tweet_clean(sentence))]

def tweet_clean(sentence):
    sentence = re.sub(r'https?://\S+', ' ', sentence)
    return sentence

tokenizer = nltk.TweetTokenizer()
stopwords_set = set(stopwords.words('english'))
TEXT = data.Field(sequential=True, tokenize=tokenize, stop_words=stopwords_set)
LABEL = data.Field(sequential=False, preprocessing=make_one_hot, dtype=torch.float, use_vocab=False)
dataset = data.TabularDataset(
    path='./train_gold.json', format='json',
    fields={'text': ('text', TEXT),
            'reply': ('reply', TEXT),
            'categories': ('categories', LABEL)})
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_set, valid_set = dataset.split(split_ratio=0.9)

MAX_VOCAB_SIZE = 50000
TEXT.build_vocab(dataset, 
                 max_size=MAX_VOCAB_SIZE, 
                 vectors="glove.6B.100d",
                 unk_init=torch.Tensor.normal_)

UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, 
                 dropout, load=False):
        
        super(CNN, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
                                    nn.Conv2d(in_channels = 1, 
                                              out_channels = n_filters, 
                                              kernel_size = (fs, embedding_dim)) 
                                    for fs in filter_sizes
                                    ])
        
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)
        
        if not load:
            self.init_weights()

    def forward(self, text):
        #text = [sent len, batch size]
        text = text.permute(1, 0)
                
        #text = [batch size, sent len]
        
        embedded = self.embedding(text)
                
        #embedded = [batch size, sent len, emb dim]
        
        embedded = embedded.unsqueeze(1)
        
        #embedded = [batch size, 1, sent len, emb dim]
        
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
            
        #conv_n = [batch size, n_filters, sent len - filter_sizes[n]]
        
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        
        #pooled_n = [batch size, n_filters]
        
        cat = self.dropout(torch.cat(pooled, dim = 1))

        #cat = [batch size, n_filters * len(filter_sizes)]
            
        return self.fc(cat)

    def init_weights(self):
        pretrained_embeddings = TEXT.vocab.vectors

        self.embedding.weight.data.copy_(pretrained_embeddings)
        self.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
        self.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
N_FILTERS = 100
FILTER_SIZES = [3, 4, 5]
OUTPUT_DIM = 43
DROPOUT = 0.5
model = CNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT)


import torch.optim as optim

criterion = nn.BCEWithLogitsLoss()

model = model.to(device)
criterion = criterion.to(device)

def categorical_accuracy(pred, label):
    batch_size = pred.size(0)
    recall = 0.
    pred = torch.sigmoid(pred)
    for i in range(batch_size):
      _, pred_topi = pred[i].topk(6)
      label_index = set((label[i] == 1.0).nonzero().view(-1).tolist())
      recall += (len(set(pred_topi.tolist()) & label_index) / len(label_index))
    return recall / batch_size

def train(model, iterator, optimizer, criterion):
    
    epoch_loss = 0
    epoch_recall = 0
    
    model.train()
    
    for batch in iterator:
        optimizer.zero_grad()
       
        predictions = model(torch.cat((batch.text, batch.reply)))
        
        loss = criterion(predictions, batch.categories)
        
        recall = categorical_accuracy(predictions, batch.categories)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_recall += recall
        
    return epoch_loss / len(iterator), epoch_recall / len(iterator)

def evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_recall = 0
    
    model.eval()
    
    with torch.no_grad():
        
        for batch in iterator:
            predictions = model(torch.cat((batch.text, batch.reply)))
            loss = criterion(predictions, batch.categories)
            
            recall = categorical_accuracy(predictions, batch.categories)

            epoch_loss += loss.item()
            epoch_recall += recall
        
    return epoch_loss / len(iterator), epoch_recall / len(iterator)

import time

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def predicts(model, path, text, min_len=5):
    model.eval()
    test_data = []

    with open(path, encoding='utf8') as f:
        test_data = f.readlines()
    test_data = [json.loads(data) for data in test_data]
    file_type = path.split('_')[0]
    outfile = 'eval.json' if  file_type == 'test' else f"{file_type}.json"
    with torch.no_grad():
        with open(outfile, 'w', encoding='utf8') as f:
            
            for idx, tweet in enumerate(test_data):
                tokenized = [token for token in tokenize(tweet['text'].lower()) if token not in stopwords_set]
                tokenized += [token for token in tokenize(tweet['reply'].lower()) if token not in stopwords_set]
                
                if len(tokenized) < min_len:
                    tokenized += ['<pad>'] * (min_len - len(tokenized))
                
                indexed = [text.vocab.stoi[token] for token in tokenized]

                input = torch.LongTensor(indexed).to(device)
                input = input.unsqueeze(1)
                
                predictions = torch.sigmoid(model(input))
                _, topi = predictions.topk(6)
                topi = topi.view(-1).cpu()

                pred_categories = [categories[cat_idx] for cat_idx in topi]
                tweet['categories'] = pred_categories
                json.dump(tweet, f, ensure_ascii=False)
                f.write('\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50, help='num of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--patience', type=int, default=5, help='patience of early stopping')
    parser.add_argument('--eval', type=str, help='load model and evaluate')
    parser.add_argument('--testfile', type=str, default='test_unlabeled.json', help='load testfile')
    parser.add_argument('--full', action='store_true', help='train with full datatset')
    args = parser.parse_args()

    epochs = args.epochs
    lr = args.lr
    batch_size = args.batch_size
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    if args.eval is None:
        valid_iterator = None
        real_train_set = None
        early_stopping = None
        train_recall = 0.
        valid_recall = 0.

        if not args.full:
            real_train_set = train_set
            valid_iterator = data.Iterator(valid_set, batch_size, train=False, sort=False, shuffle=True, device=device)
            early_stopping = EarlyStopping(patience=args.patience, verbose=True)
        else:
            real_train_set = dataset
        
        train_iterator = data.Iterator(real_train_set, batch_size, train=True, shuffle=True, device=device)
        
        for epoch in range(epochs):

            start_time = time.time()

            train_loss, train_recall = train(model, train_iterator, optimizer, criterion)
            
            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            
            print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Recall: {train_recall}')
            
            if not args.full:
                valid_loss, valid_recall= evaluate(model, valid_iterator, criterion)
                print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Recall: {valid_recall}')
                
                early_stopping(valid_loss, model)

                if early_stopping.early_stop:
                    print(f"Epochs: {epoch} - Early Stopping...")
                    break    
        torch.save(
            {
                'net': model.state_dict(),
                'dim': INPUT_DIM,
                'text': TEXT
            }, f'cnn-model-vocab-{INPUT_DIM}-recall-{train_recall}.pth')
    else:
        model = torch.load(args.eval)
        input_dim = model['dim']
        text = model['text']
        load_model = CNN(input_dim, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, load=True).to(device)
        load_model.load_state_dict(model['net'])
        
        eval_set = data.TabularDataset(
            path='./train_gold.json', format='json',
            fields={'text': ('text', text),
                    'reply': ('reply', text),
                    'categories': ('categories', LABEL)})
        eval_iterator = data.Iterator(eval_set, 128, train=False, sort=False, shuffle=True, device=device)
        _, recall = evaluate(load_model, eval_iterator, criterion)
        print(f'Recall: {recall}')
        predicts(load_model, args.testfile, text)
