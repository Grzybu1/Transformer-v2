from transformer import Transformer
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import numpy as np

def is_end_token(sentence):
    for curr_token_id in sentence:
        if curr_token_id == 1: #pad
            return False
        elif curr_token_id == 2: #eos
            return True
    raise OverflowError

def first_pad_index_pos(sentence):
    for i, curr_token_id in enumerate(sentence):
        if curr_token_id == 1: #pad
            return i
    raise OverflowError


embedding_size = 128
num_heads = 8
num_layers = 3
d_hidden = 516
max_seq_size = 75
batch_size = 50
data_size = 500
states_file = 'trained/trained_translation_transformer_2.pt'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()
device = 'cpu'
torch.set_default_device(device)

tokenizer_en = AutoTokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt", src_lang="en_XX")
tokenizer_fr = AutoTokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt", src_lang="fr_XX")
en_vocab_size = tokenizer_en.vocab_size
fr_vocab_size = tokenizer_fr.vocab_size

# print(tokenizer_en.bos_token) #0
# print(tokenizer_en.pad_token) #1
# print(tokenizer_en.eos_token) #2

model = Transformer(src_vocab_size=en_vocab_size, tgt_vocab_size=fr_vocab_size, embedding_size=embedding_size, num_heads=num_heads, num_layers=num_layers, max_seq_len=max_seq_size)
model.load_state_dict(torch.load(states_file))
input_sentence = ['Is Tom OK?']
# tokenized_output = [1 for _ in range(max_seq_size)]
# tokenized_output[0] = 0
tokenized_output = [0, 1]
tokenized_output = torch.tensor(tokenized_output, dtype=int, device=device).unsqueeze(0)
tokenized_input  = torch.tensor(tokenizer_en(list(input_sentence),padding='max_length', max_length=max_seq_size, verbose = True)['input_ids'], dtype=int, device=device)


# try:
#     while not is_end_token(tokenized_output[0]):
#         # print(model(tokenized_input, tokenized_output).shape)
#         out = model(tokenized_input, tokenized_output)
#         print(first_pad_index_pos(tokenized_output[0])-1)
#         out = out[0, first_pad_index_pos(tokenized_output[0])-1]
#         out_probs = F.softmax(out, dim=0)
#         index_max =  torch.argmax(out_probs)
#         print(index_max)
#         tokenized_output[0, first_pad_index_pos(tokenized_output[0])] = index_max
# finally:    
#     print(tokenizer_fr.decode(tokenized_output[0]))

print(torch.argmax(model(tokenized_input, tokenized_output)[0][1]))