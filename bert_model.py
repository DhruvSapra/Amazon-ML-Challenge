import torch 
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader,SubsetRandomSampler
import torch.optim as optim

import os
import copy
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pylab import rcParams
import csv
import time
from tqdm import tqdm
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from transformers import AutoTokenizer,AutoModel,AutoModelForSequenceClassification,AdamW,get_linear_schedule_with_warmup

seed_val = 42 
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

if torch.cuda.is_available():
    device = torch.device('cuda')
    print("Running on gpu",torch.cuda.get_device_name(0))
else:
    device = 'cpu'
    print('No GPU found Running on cpu')

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
sample_df = pd.read_csv("sample_submission.csv")

train_df['TITLE'].isnull().sum()

len(train_df)-len(train_df.drop_duplicates())

train_df.shape

train_df = train_df.drop_duplicates()
train_df.shape

train_df = train_df[train_df['TITLE'].notnull()]

train_df['TITLE'].isnull().sum()

le = LabelEncoder()
train_df['PRODUCT_LENGTH'] = le.fit_transform(train_df['PRODUCT_LENGTH'])
train_df['PRODUCT_LENGTH'].max()

sentences = train_df['TITLE'].values 
labels = train_df['PRODUCT_LENGTH'].values

print(sentences.shape,labels.shape)

train_sentences,val_sentences,train_labels,val_labels = train_test_split(sentences,labels,test_size = 0.1,random_state=seed_val)

print(f"No. of training sentences {len(train_sentences)}")
print(f"No. of validation sentences {len(val_sentences)}")

train_df.memory_usage(deep= True)*(1e-6)

indices , cnts = np.unique(labels,return_counts=True)

sns.countplot(y = cnts[ (cnts >=10) & (cnts <=100)] )

model_name = 'bert-base-multilingual-cased'
max_input_length = 128
batch_size = 64

tokenizer = AutoTokenizer.from_pretrained(model_name)

idx = 100000
sample_text = sentences[idx]
tokens =tokenizer.tokenize(sample_text)
token_ids = tokenizer.convert_tokens_to_ids(tokens)
print('Sample text {}'.format(sample_text))
print('Tokens {}'.format(tokens))
print('Token IDS {}'.format(token_ids))

tokenizer.sep_token,tokenizer.sep_token_id

tokenizer.cls_token,tokenizer.cls_token_id

tokenizer.pad_token,tokenizer.pad_token_id

tokenizer.unk_token,tokenizer.unk_token_id

encoding = tokenizer.encode_plus(
    sample_text,
    max_length = max_input_length,
    add_special_tokens = True,
    pad_to_max_length=True,
    return_attention_mask = True,
    return_token_type_ids = False,
    return_tensors = 'pt'
)

encoding

encoding.keys()

base_model = AutoModel.from_pretrained(model_name)

base_model(**encoding)['pooler_output']

class AmazonDataset(Dataset):

  def __init__(self, sentences, labels, tokenizer, max_length,with_labels=True):
    self.sentences = sentences
    self.labels = labels
    self.tokenizer = tokenizer
    self.max_length = max_length
    self.with_labels = with_labels
  
  def __len__(self):
    return len(self.sentences)
  
  def __getitem__(self, idx):
    sentence = str(self.sentences[idx])
    encoding = self.tokenizer.encode_plus(
      sentence,
      add_special_tokens=True,
      max_length=self.max_length,
      return_token_type_ids=False,
      pad_to_max_length=True,
      return_attention_mask=True,
      return_tensors='pt',
    )

    if self.with_labels:
        
        label = self.labels[idx]

        return {
            'sentence': sentence,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
    else:
        return {
            'sentence': sentence,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }

def create_data_loaders(sentences,labels,tokenizer,max_input_length,batch_size,with_labels):
    ds = AmazonDataset(
        sentences =sentences,
        labels=labels,
        tokenizer=tokenizer,
        max_length=max_input_length,
        with_labels = with_labels
    )

    return DataLoader(
        ds,
        batch_size=batch_size
    )

train_loader = create_data_loaders(
    train_sentences,
    train_labels,
    tokenizer,
    max_input_length=max_input_length,
    batch_size=batch_size,
    with_labels = True
)

val_loader = create_data_loaders(
    val_sentences,
    val_labels,
    tokenizer,
    max_input_length=max_input_length,
    batch_size=batch_size,
    with_labels = True
)

class AmazonClassifier(nn.Module):

  def __init__(self,base_model_name, n_classes):
    super(AmazonClassifier, self).__init__()
    self.base_model = AutoModel.from_pretrained(base_model_name)
    self.drop = nn.Dropout(p=0.3)
    self.out = nn.Linear(self.base_model.config.hidden_size, n_classes)
  
  def forward(self, input_ids, attention_mask):
    pooled_output = self.base_model(
      input_ids=input_ids,
      attention_mask=attention_mask
    )['pooler_output']
    output = self.drop(pooled_output)
    return self.out(output)

model = AmazonClassifier(base_model_name=model_name,n_classes=len(np.unique(labels)))

model.to(device)

num_epochs = 2

optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)

total_steps = len(train_loader) * num_epochs

scheduler = get_linear_schedule_with_warmup(
  optimizer,
  num_warmup_steps=0,
  num_training_steps=total_steps
)

loss_fn = nn.CrossEntropyLoss().to(device)

def train_epoch(
  model, 
  data_loader, 
  loss_fn, 
  optimizer, 
  device, 
  scheduler, 
  n_examples
):
  model = model.train()

  losses = []
  correct_predictions = 0
  
  for i,d in enumerate(data_loader):
    if i%100 == 0:
        print(f"Processing batch {i+1}/{len(data_loader)}")
    input_ids = d["input_ids"].to(device)
    attention_mask = d["attention_mask"].to(device)
    labels = d["labels"].to(device)

    outputs = model(
      input_ids=input_ids,
      attention_mask=attention_mask
    )

    _, preds = torch.max(outputs, dim=1)
    loss = loss_fn(outputs, labels)

    correct_predictions += torch.sum(preds == labels)
    losses.append(loss.item())

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

  return correct_predictions.double() / n_examples, np.mean(losses)

import torch
import numpy as np
from collections import defaultdict

def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()

    losses = []
    squared_errors = []

    with torch.no_grad():
        for i,d in enumerate(data_loader):
            if i % 100 == 0:
                print(f"Processing batch {i+1}/{len(data_loader)}")
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            preds = outputs.squeeze()

            loss = loss_fn(outputs, labels)

            squared_errors.extend((preds - labels)**2)
            losses.append(loss.item())

    rmse = torch.sqrt(torch.mean(torch.tensor(squared_errors)))
    return -rmse, np.mean(losses)  # return negative RMSE for optimization

def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    model = model.train()

    losses = []
    correct_predictions = 0

    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        loss = loss_fn(outputs.squeeze(), labels)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        scheduler.step()

    return correct_predictions.double() / n_examples, np.mean(losses)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
history = defaultdict(list)
best_rmse = float('inf')

for epoch in tqdm(range(num_epochs)):

    print(f'Epoch {epoch + 1}/{num_epochs}')
    print('-' * 10)

    train_acc, train_loss = train_epoch(
        model,
        train_loader,
        loss_fn,
        optimizer,
        device,
        scheduler,
        len(train_labels)
    )

    print(f'Train loss {train_loss} accuracy {train_acc}')

    val_acc, val_loss = eval_model(
        model,
        val_loader,
        loss_fn,
        device,
        len(val_labels)
    )

    print(f'Val loss {val_loss} accuracy {val_acc}')
    print()

    history['train_acc'].append(train_acc)
    history['train_loss'].append(train_loss)
    history['val_acc'].append(-val_acc)  # append negative RMSE for optimization
    history['val_loss'].append(val_loss)

    if val_acc < best_rmse:  # use < instead of > since we're using negative RMSE
        torch.save(model.state_dict(), 'best_model_state.bin')
        best_rmse = val_acc
        print(best_rmse)
