import numpy as np
from sklearn.model_selection import train_test_split
import time
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from transformer import Transformer
from transformers import AutoTokenizer

def batchify(data, batch_size):
    l = len(data)
    for ndx in range(0, l, batch_size):
        yield data[ndx:min(ndx + batch_size, l)]
    return data

if __name__ == "__main__":
    embedding_size = 128
    num_heads = 8
    num_layers = 3
    d_hidden = 516
    max_seq_size = 75
    batch_size = 50
    data_size = 50000
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print()
    torch.set_default_device(device)
    
    tokenizer_en = AutoTokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt", src_lang="en_XX")
    tokenizer_fr = AutoTokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt", src_lang="fr_XX")
    en_vocab_size = tokenizer_en.vocab_size
    fr_vocab_size = tokenizer_fr.vocab_size
    if os.path.isfile("x_train.txt") & os.path.isfile("x_test.txt") & os.path.isfile("y_train.txt") & os.path.isfile("y_test.txt"):
        x_train = np.loadtxt("x_train.txt", dtype=str, delimiter='&')
        x_test = np.loadtxt("x_test.txt", dtype=str, delimiter='&')
        y_train = np.loadtxt("y_train.txt", dtype=str, delimiter='&')
        y_test = np.loadtxt("y_test.txt", dtype=str, delimiter='&')
    else:
        data = np.loadtxt("datasets/eng_fr_corrected.csv",
                    delimiter=",", dtype=str, skiprows=1) 
        np.random.shuffle(data)
        data = data[:data_size]
        
        data_eng, data_fr = data[:data_size].T
        x_train, x_test, y_train, y_test = train_test_split(data_eng, data_fr, test_size=0.2, random_state=44)
        np.savetxt("x_train.txt", x_train, fmt="%s")
        np.savetxt("x_test.txt", x_test, fmt="%s")
        np.savetxt("y_train.txt", y_train, fmt="%s")
        np.savetxt("y_test.txt", y_test, fmt="%s")
    
    
    # print(tokenizer_en.bos_token) #0
    # print(tokenizer_en.pad_token) #1
    # print(tokenizer_en.eos_token) #2
    
    tokenized_x_train = torch.tensor(tokenizer_en(list(x_train),padding='max_length', max_length=max_seq_size, verbose = True)['input_ids'], dtype=int, device=device)
    tokenized_x_test  = torch.tensor(tokenizer_en(list(x_test),padding='max_length', max_length=max_seq_size, verbose = True)['input_ids'], dtype=int, device=device)
    tokenized_y_train = torch.tensor(tokenizer_fr(list(y_train),padding='max_length', max_length=max_seq_size, verbose = True)['input_ids'], dtype=int, device=device)
    tokenized_y_test = torch.tensor(tokenizer_fr(list(y_test),padding='max_length', max_length=max_seq_size, verbose = True)['input_ids'], dtype=int, device=device)
    
    #set bos tokens instead of class tokens
    tokenized_x_train[:,0] = 0
    tokenized_x_test[:,0] = 0
    tokenized_y_train[:,0] = 0
    tokenized_y_test[:,0] = 0
    
    model = Transformer(src_vocab_size=en_vocab_size, tgt_vocab_size=fr_vocab_size, embedding_size=embedding_size, num_heads=num_heads, num_layers=num_layers, max_seq_len=max_seq_size)
    file_to_start = 'trained_translation_transformer.pt'
    if file_to_start != '':
        model.load_state_dict(torch.load(file_to_start))
        
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    model.train()
print(f"There are: {tokenized_x_train.shape[0]//batch_size} batches")
start_time = time.time()
for epoch in range(10):
    epoch_start_time = time.time()
    print(f"Epoch: {epoch}")
    batch = 0
    for x_train_batch, y_train_batch in zip(batchify(tokenized_x_train, batch_size), batchify(tokenized_y_train, batch_size)):
        optimizer.zero_grad()
        output = model(x_train_batch, y_train_batch)
        loss = criterion(output.contiguous().view(-1, fr_vocab_size), y_train_batch.contiguous().view(-1))
        loss.backward()
        optimizer.step()
        print(f"Batch: {batch}, Loss: {loss.item()}")
        batch = batch+1
    print(f"Epoch took: {(time.time() - epoch_start_time)} s")
    torch.save(model.state_dict(), f'trained_translation_transformer_epoch_{epoch}.pt')

print(f"Learning took: {(time.time() - epoch_start_time)/3600} h")
torch.save(model.state_dict(), 'trained_translation_transformer.pt')
        