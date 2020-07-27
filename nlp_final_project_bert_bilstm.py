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
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

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
    tokens = tokenizer.tokenize(tweet_clean(sentence))
    tokens = tokens[:(max_input_length-2)]
    return tokens

def tweet_clean(sentence):
    sentence = re.sub(r'https?://\S+', ' ', sentence)
    return sentence

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
CLS_IDX = tokenizer.cls_token_id
UNK_IDX = tokenizer.unk_token_id
PAD_IDX = tokenizer.pad_token_id
EOS_IDX = tokenizer.sep_token_id
max_input_length = tokenizer.max_model_input_sizes['bert-base-uncased']
stopwords_set = set(stopwords.words('english'))

TEXT = data.Field(batch_first=True, tokenize=tokenize, 
                stop_words=stopwords_set, use_vocab=False, preprocessing=tokenizer.convert_tokens_to_ids, 
                init_token=CLS_IDX, eos_token=EOS_IDX, pad_token=PAD_IDX,
                unk_token=UNK_IDX)
LABEL = data.Field(sequential=False, preprocessing=make_one_hot, dtype=torch.float, use_vocab=False)

dataset = data.TabularDataset(
    path='./train_gold.json', format='json',
    fields={'text': ('text', TEXT),
            'reply': ('reply', TEXT),
            'categories': ('categories', LABEL)})

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_set, valid_set = dataset.split(split_ratio=0.9)

import torch.nn as nn
import torch.nn.functional as F

class BertbiLSTM(nn.Module):
    def __init__(self, bert, n_layers, bidirectional, 
            output_dim, dropout, hidden_dim=None, 
            freeze=True):
        super(BertbiLSTM, self).__init__()
        self.bert = bert
        self.freeze = freeze
        self.bidirectional = bidirectional
        self.n_layers = n_layers
        embedding_dim = self.bert.config.to_dict()['hidden_size']
        self.lstm = nn.LSTM(embedding_dim, hidden_size=embedding_dim if hidden_dim is None else hidden_dim, 
                    num_layers=n_layers, bidirectional=bidirectional, batch_first=True, 
                    dropout=0 if n_layers < 2 else dropout)
        lstm_input_dim = embedding_dim if hidden_dim is None else hidden_dim
        self.fc = nn.Linear(2 * lstm_input_dim if bidirectional else lstm_input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
        if freeze:
            self.freeze_bert()

    def forward(self, text):
        #text = [batch size, sent_length]
        embedded = None
        
        if self.freeze:
            with torch.no_grad():
                embedded = self.bert(text)[0]
        else:
            embedded = self.bert(text)[0]
        _, (hidden, cell) = self.lstm(embedded)
        
        #embedded = [batch size, sent len, emb dim]
        if self.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        else:
            hidden = self.dropout(hidden[-1,:,:])

        return self.fc(hidden)
    
    def freeze_bert(self):
        for name, param in self.named_parameters():
            if name.startswith('bert'):
                param.requires_grad = False

OUTPUT_DIM = 43
bert = BertModel.from_pretrained('bert-base-uncased')

import torch.optim as optim

criterion = nn.BCEWithLogitsLoss()
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
    
    for batch in tqdm(iterator):
        optimizer.zero_grad()
       
        predictions = model(torch.cat((batch.text, batch.reply), dim=1))
        
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
        
        for batch in tqdm(iterator):
            predictions = model(torch.cat((batch.text, batch.reply), dim=1))
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

def predicts(model, path):
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
                tokens = [token for token in tokenize(tweet['text'].lower()) if token not in stopwords_set]
                tokens += [token for token in tokenize(tweet['reply'].lower()) if token not in stopwords_set]
                
                tokens = tokens[:(max_input_length-2)]
                indexed = [CLS_IDX] + tokenizer.convert_tokens_to_ids(tokens) + [EOS_IDX]
                input = torch.LongTensor(indexed).to(device)
                input = input.unsqueeze(0)
                
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
    parser.add_argument('--lr', type=float, default=.001, help='learning rate')
    parser.add_argument('--patience', type=int, default=5, help='patience of early stopping')
    parser.add_argument('--eval', type=str, help='load model and evaluate')
    parser.add_argument('--testfile', type=str, default='test_unlabeled.json', help='load testfile')
    parser.add_argument('--dropout', type=float, default=.25, help='dropout rate')
    parser.add_argument('--full', action='store_true', help='train with full datatset')
    args = parser.parse_args()

    model_type = 'bertbilstm'
    
    model = BertbiLSTM(bert, n_layers=2, hidden_dim=256, output_dim=OUTPUT_DIM, bidirectional=True, dropout=args.dropout)
    
    model = model.to(device)
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
            }, f'{model_type}-model-recall-{train_recall}.pth')
    else:
        model = torch.load(args.eval)
        load_model = None

        load_model = BertbiLSTM(bert, n_layers=2, hidden_dim=256, output_dim=OUTPUT_DIM, bidirectional=True, dropout=args.dropout)
       
        load_model.load_state_dict(model['net'])
        load_model = load_model.to(device) 

        eval_set = data.TabularDataset(
            path='./train_gold.json', format='json',
            fields={'text': ('text', TEXT),
                    'reply': ('reply', TEXT),
                    'categories': ('categories', LABEL)})
        eval_iterator = data.Iterator(eval_set, 128, train=False, sort=False, shuffle=True, device=device)
        _, recall = evaluate(load_model, eval_iterator, criterion)
        print(f'Recall: {recall}')
        predicts(load_model, args.testfile)
